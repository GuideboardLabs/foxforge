#!/bin/bash
# Build Ollama ROCm GPU library for gfx1010 (RX 5700 XT)
# Fixes SDMA ring crash caused by gfx1030 shaders on gfx1010 hardware

set -e

OLLAMA_VERSION="v0.18.0"
BUILD_DIR="$HOME/ollama-build"
GO_VERSION="1.23.5"
GO_TARBALL="go${GO_VERSION}.linux-amd64.tar.gz"
GO_URL="https://go.dev/dl/${GO_TARBALL}"
CMAKE_BUILD_DIR="$BUILD_DIR/build"

echo "=== Ollama gfx1010 Build Script ==="
echo "This will take 15-30 minutes to compile HIP kernels."
echo ""

# --- Step 1: Go ---
if ! /usr/local/go/bin/go version &>/dev/null; then
    echo "[1/5] Installing Go ${GO_VERSION}..."
    cd /tmp
    wget -q --show-progress "$GO_URL"
    sudo tar -C /usr/local -xzf "$GO_TARBALL"
    rm "$GO_TARBALL"
else
    echo "[1/5] Go already installed: $(/usr/local/go/bin/go version)"
fi
export PATH=$PATH:/usr/local/go/bin

# --- Step 2: Build deps ---
echo "[2/5] Installing build dependencies..."
sudo apt install -y cmake build-essential

# --- Step 3: Clone Ollama (reuse if exists) ---
if [ ! -d "$BUILD_DIR/.git" ]; then
    echo "[3/5] Cloning Ollama ${OLLAMA_VERSION}..."
    rm -rf "$BUILD_DIR"
    git clone --depth=1 --branch "$OLLAMA_VERSION" https://github.com/ollama/ollama.git "$BUILD_DIR"
else
    echo "[3/5] Using existing clone at $BUILD_DIR"
fi

# --- Step 4: Build ROCm GPU library via cmake ---
echo "[4/5] Building ROCm GPU library with gfx1010 (slow)..."
mkdir -p "$CMAKE_BUILD_DIR"
cd "$CMAKE_BUILD_DIR"

cmake "$BUILD_DIR" \
    --preset "ROCm 6" \
    -DAMDGPU_TARGETS=gfx1010 \
    -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 \
    -DCMAKE_INSTALL_PREFIX="$CMAKE_BUILD_DIR/install"

cmake --build . --config Release --parallel $(nproc) --target install

# --- Step 5: Install into Ollama ---
echo "[5/5] Installing GPU libraries into Ollama..."
sudo systemctl stop ollama

# Back up old libs
sudo cp -r /usr/local/lib/ollama/rocm /usr/local/lib/ollama/rocm.bak 2>/dev/null || true
sudo cp /usr/local/bin/ollama /usr/local/bin/ollama.bak

# Install new libs
INSTALL_ROCM=$(find "$CMAKE_BUILD_DIR/install" -type d -name "rocm" 2>/dev/null | head -1)
if [ -n "$INSTALL_ROCM" ]; then
    echo "  Copying new ROCm libs from $INSTALL_ROCM..."
    sudo cp -r "$INSTALL_ROCM"/. /usr/local/lib/ollama/rocm/
else
    echo "  Copying from lib/ollama/rocm directly..."
    INSTALL_LIB=$(find "$CMAKE_BUILD_DIR/install" -name "libggml-hip.so" 2>/dev/null | head -1)
    if [ -n "$INSTALL_LIB" ]; then
        sudo cp "$(dirname $INSTALL_LIB)"/*.so /usr/local/lib/ollama/rocm/
    else
        echo "ERROR: Could not find built libggml-hip.so. Check cmake output above."
        exit 1
    fi
fi

# Build and install the Ollama binary
echo "  Building Ollama binary..."
cd "$BUILD_DIR"
/usr/local/go/bin/go build -o ollama .
sudo cp ollama /usr/local/bin/ollama

# Fix service drop-in: remove bad gfx1030 override
sudo tee /etc/systemd/system/ollama.service.d/rocm.conf > /dev/null << 'EOF'
[Service]
Environment="ROCR_VISIBLE_DEVICES=0"
EOF

sudo systemctl daemon-reload
sudo systemctl start ollama

echo ""
echo "=== Done ==="
echo "Checking Ollama GPU detection (waiting 5s)..."
sleep 5
journalctl -u ollama -n 10 --no-pager | grep -E "(compute|inference|ROCm|gfx|error|WARN|crash)"
