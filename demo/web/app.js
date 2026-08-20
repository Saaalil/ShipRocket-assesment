const SAMPLE_RATE = 16000;
const MAX_SECONDS = 8;
const N_MELS = 80;
const N_FRAMES = 800;

const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const thresholdEl = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");
const recordBtn = document.getElementById("record");
const stopBtn = document.getElementById("stop");
const fileEl = document.getElementById("file");

let session = null;
let mediaRecorder = null;
let chunks = [];

thresholdEl.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdEl.value).toFixed(2);
});

async function loadModel() {
  statusEl.textContent = "Loading ONNX model…";
  const localUrl = "../../artifacts/model_int8.onnx";
  const fallbackUrl =
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx";
  try {
    session = await ort.InferenceSession.create(localUrl, { executionProviders: ["wasm"] });
  } catch {
    session = await ort.InferenceSession.create(fallbackUrl, { executionProviders: ["wasm"] });
  }
  statusEl.textContent = "Model ready. Record or upload a clip after a pause.";
}

function toMono(buffer) {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  return left.map((value, index) => (value + right[index]) * 0.5);
}

function resample(audio, fromRate) {
  if (fromRate === SAMPLE_RATE) return audio;
  const duration = audio.length / fromRate;
  const length = Math.max(1, Math.round(duration * SAMPLE_RATE));
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const src = (i / length) * audio.length;
    const left = Math.floor(src);
    const right = Math.min(left + 1, audio.length - 1);
    const frac = src - left;
    output[i] = audio[left] * (1 - frac) + audio[right] * frac;
  }
  return output;
}

function leftPadLast8s(audio) {
  const maxSamples = SAMPLE_RATE * MAX_SECONDS;
  const sliced = audio.length > maxSamples ? audio.slice(audio.length - maxSamples) : audio;
  if (sliced.length === maxSamples) return sliced;
  const padded = new Float32Array(maxSamples);
  padded.set(sliced, maxSamples - sliced.length);
  return padded;
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function logMel(audio) {
  // Approximate Whisper log-mel for the static demo. Python export remains the reference.
  const nFft = 400;
  const hop = 160;
  const spec = [];
  for (let frame = 0; frame < N_FRAMES; frame += 1) {
    const start = frame * hop;
    const windowed = new Float32Array(nFft);
    for (let i = 0; i < nFft && start + i < audio.length; i += 1) {
      const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (nFft - 1));
      windowed[i] = audio[start + i] * hann;
    }
    const mag = new Float32Array(nFft / 2 + 1);
    for (let k = 0; k < mag.length; k += 1) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < nFft; n += 1) {
        const angle = (2 * Math.PI * k * n) / nFft;
        re += windowed[n] * Math.cos(angle);
        im -= windowed[n] * Math.sin(angle);
      }
      mag[k] = re * re + im * im;
    }
    spec.push(mag);
  }
  const melMin = hzToMel(0);
  const melMax = hzToMel(SAMPLE_RATE / 2);
  const filters = [];
  for (let m = 0; m < N_MELS; m += 1) {
    const left = melToHz(melMin + ((melMax - melMin) * m) / (N_MELS + 1));
    const center = melToHz(melMin + ((melMax - melMin) * (m + 1)) / (N_MELS + 1));
    const right = melToHz(melMin + ((melMax - melMin) * (m + 2)) / (N_MELS + 1));
    filters.push({ left, center, right });
  }
  const features = new Float32Array(N_MELS * N_FRAMES);
  for (let t = 0; t < N_FRAMES; t += 1) {
    const mag = spec[t];
    for (let m = 0; m < N_MELS; m += 1) {
      const { left, center, right } = filters[m];
      let energy = 0;
      for (let k = 0; k < mag.length; k += 1) {
        const hz = (k * SAMPLE_RATE) / nFft;
        let weight = 0;
        if (hz >= left && hz <= center) weight = (hz - left) / (center - left || 1);
        else if (hz > center && hz <= right) weight = (right - hz) / (right - center || 1);
        energy += mag[k] * weight;
      }
      features[m * N_FRAMES + t] = Math.log10(Math.max(energy, 1e-10));
    }
  }
  return features;
}

async function decodeBlob(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
  await ctx.close();
  return leftPadLast8s(resample(toMono(decoded), decoded.sampleRate));
}

async function infer(audio) {
  if (!session) await loadModel();
  const features = logMel(audio);
  const tensor = new ort.Tensor("float32", features, [1, N_MELS, N_FRAMES]);
  const started = performance.now();
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const elapsed = performance.now() - started;
  const probability = outputs[session.outputNames[0]].data[0];
  const threshold = Number(thresholdEl.value);
  resultEl.textContent = JSON.stringify(
    {
      label: probability >= threshold ? "COMPLETE" : "INCOMPLETE",
      probability_complete: Number(probability.toFixed(4)),
      threshold,
      inference_ms: Number(elapsed.toFixed(1)),
      note: "Python Gradio demo is the numerically matched reference.",
    },
    null,
    2
  );
}

recordBtn.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (event) => chunks.push(event.data);
  mediaRecorder.start();
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "Recording… speak, pause, then stop.";
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach((track) => track.stop());
  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    const audio = await decodeBlob(blob);
    await infer(audio);
    recordBtn.disabled = false;
    statusEl.textContent = "Ready.";
  };
});

fileEl.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const audio = await decodeBlob(file);
  await infer(audio);
});

loadModel().catch((error) => {
  statusEl.textContent = `Could not load model: ${error}`;
});
