/* FFmpeg Web Worker — runs ffmpeg.wasm off the main thread */
/* Uses ffmpeg.wasm 0.11.8 single-thread mode (no SharedArrayBuffer needed) */

let ffmpeg = null;
let ffmpegLoaded = false;
let pendingTasks = {};
let taskIdCounter = 0;

self.onmessage = async function(e) {
  const { type, id, cmd, files, taskName } = e.data;

  try {
    if (type === 'load') {
      if (!ffmpegLoaded) {
        const { createFFmpeg, fetchFile } = await import(
          'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.8/dist/ffmpeg.min.js'
        );
        ffmpeg = createFFmpeg({ log: false });
        await ffmpeg.load();
        ffmpegLoaded = true;
      }
      self.postMessage({ type: 'loaded', id });
      return;
    }

    if (type === 'cancel') {
      // ffmpeg.wasm 0.11.8 doesn't support true cancellation
      // We mark the task as cancelled; the main thread ignores results
      if (pendingTasks[id]) {
        pendingTasks[id].cancelled = true;
        delete pendingTasks[id];
      }
      return;
    }

    if (type === 'run') {
      if (!ffmpegLoaded) {
        const { createFFmpeg, fetchFile } = await import(
          'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.8/dist/ffmpeg.min.js'
        );
        ffmpeg = createFFmpeg({ log: false });
        await ffmpeg.load();
        ffmpegLoaded = true;
      }

      const task = { cancelled: false };
      pendingTasks[id] = task;

      // Write input files
      for (const f of (files || [])) {
        ffmpeg.FS('writeFile', f.name, new Uint8Array(f.data));
      }

      // Set up progress
      ffmpeg.setProgress(({ ratio }) => {
        if (pendingTasks[id] && !pendingTasks[id].cancelled) {
          self.postMessage({ type: 'progress', id, ratio: ratio || 0 });
        }
      });

      // Run command
      const output = await ffmpeg.run(...cmd);

      // Check cancellation
      if (task.cancelled) {
        self.postMessage({ type: 'cancelled', id });
        return;
      }

      // Read output files (all .mp4, .srt, .wav, etc. in output)
      const outputFiles = [];
      const FS = ffmpeg.FS;
      // Try to find output by checking common extensions
      const outName = cmd[cmd.length - 1];
      try {
        const data = FS('readFile', outName);
        outputFiles.push({
          name: outName,
          data: data.buffer.slice(0) // Transferable copy
        });
      } catch (e) {
        // List all files in FS to find outputs
        const allFiles = FS('ls', '/');
        for (const f of allFiles) {
          if (f.name !== 'input' && f.name !== 'list.txt' && !f.name.startsWith('input')) {
            try {
              const d = FS('readFile', f.name);
              outputFiles.push({ name: f.name, data: d.buffer.slice(0) });
            } catch (ex) { /* skip */ }
          }
        }
      }

      if (task.cancelled) {
        self.postMessage({ type: 'cancelled', id });
        return;
      }

      delete pendingTasks[id];
      self.postMessage({ type: 'result', id, files: outputFiles },
        outputFiles.map(f => f.data).filter(d => d instanceof ArrayBuffer));

    }
  } catch (err) {
    if (pendingTasks[id]) delete pendingTasks[id];
    self.postMessage({ type: 'error', id, message: err.message || String(err) });
  }
};
