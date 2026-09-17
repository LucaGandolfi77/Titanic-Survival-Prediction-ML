export async function sharePhoto(options) {
  if (!navigator.share) throw new Error('Web Share API not available');
  return navigator.share(options);
}

export async function downloadPhoto(url, filename = 'snap.jpg') {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function canShareFiles(files) {
  return navigator.canShare && navigator.canShare({ files });
}
