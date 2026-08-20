class SilentProcessor extends AudioWorkletProcessor {
  process() {
    return true;
  }
}

registerProcessor("silent-processor", SilentProcessor);
