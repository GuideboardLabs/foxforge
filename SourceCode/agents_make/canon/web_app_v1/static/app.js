/** Canonical Vue 3 application shell with slot regions for feature code. */

const { createApp, ref, reactive, computed, onMounted, watch } = Vue;

async function apiFetch(path, options = {}) {
  const opts = { ...options };
  if (opts.body && typeof opts.body !== "string") {
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    opts.body = JSON.stringify(opts.body);
  }
  const response = await fetch(path, opts);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_err) {
      payload = {};
    }
  }
  if (!response.ok) {
    const errorBlock = payload && payload.error ? payload.error : { code: "HTTP_ERROR", message: text || response.statusText, details: null };
    const err = new Error(errorBlock.message || "Request failed");
    err.code = errorBlock.code || "HTTP_ERROR";
    err.details = errorBlock.details || null;
    err.status = response.status;
    throw err;
  }
  if (payload && payload.error) {
    const err = new Error(payload.error.message || "Request failed");
    err.code = payload.error.code || "HTTP_ERROR";
    err.details = payload.error.details || null;
    throw err;
  }
  if (payload && Object.prototype.hasOwnProperty.call(payload, "item")) {
    return payload.item;
  }
  if (payload && Object.prototype.hasOwnProperty.call(payload, "items")) {
    return payload.items;
  }
  return payload;
}

createApp({
  setup() {
    const loading = ref(false);
    const error = ref("");

    const stateBindings = { loading, error };
    const methodBindings = {};
    const computedBindings = {};

    // region: state
    const records = ref([]);
    stateBindings.records = records;
    // endregion: state

    // region: methods
    async function loadRecords() {
      loading.value = true;
      error.value = "";
      try {
        records.value = await apiFetch("/api/health");
      } catch (err) {
        error.value = String(err && err.message ? err.message : err);
      } finally {
        loading.value = false;
      }
    }
    methodBindings.loadRecords = loadRecords;
    // endregion: methods

    // region: computed
    const hasError = computed(() => Boolean(error.value));
    computedBindings.hasError = hasError;
    // endregion: computed

    // region: on-mounted
    onMounted(loadRecords);
    // endregion: on-mounted

    return {
      ...stateBindings,
      ...methodBindings,
      ...computedBindings,
    };
  },
}).mount("#app");
