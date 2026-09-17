export async function captureToBlob(canvasSource, overlayCanvas) {
  const canvas = document.createElement('canvas');
  canvas.width = canvasSource.width || canvasSource.canvas?.width || window.innerWidth;
  canvas.height = canvasSource.height || canvasSource.canvas?.height || window.innerHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(canvasSource, 0, 0, canvas.width, canvas.height);

  if (overlayCanvas) {
    ctx.drawImage(overlayCanvas, 0, 0, canvas.width, canvas.height);
  }

  return new Promise((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', 0.75);
  });
}

export async function savePhotoLocally(blob, filename = 'snap.jpg') {
  if ('showSaveFilePicker' in window) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: 'JPEG Image', accept: { 'image/jpeg': ['.jpg'] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return { saved: true, handle };
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('File System Access failed, falling back to download:', error);
      }
    }
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  return { saved: true };
}
