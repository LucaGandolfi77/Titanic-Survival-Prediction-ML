// Utility functions used across the game
function lerp(a, b, t){ return a + (b - a) * t; }

function clamp(v, min, max){ return Math.max(min, Math.min(max, v)); }

function randFloat(min=0, max=1){ return Math.random() * (max - min) + min; }

function randInt(min, max){ // inclusive
	min = Math.ceil(min); max = Math.floor(max);
	return Math.floor(Math.random() * (max - min + 1)) + min;
}

function hexToRgb(hex){
	if (!hex) return {r:0,g:0,b:0};
	hex = hex.replace('#','');
	if (hex.length === 3){ hex = hex.split('').map(ch=>ch+ch).join(''); }
	const int = parseInt(hex,16);
	return { r: (int>>16)&255, g: (int>>8)&255, b: int&255 };
}

function _componentToHex(c){ const h = c.toString(16); return h.length==1? '0'+h : h; }

function lerpColor(c1, c2, t){
	const a = hexToRgb(c1), b = hexToRgb(c2);
	const r = Math.round(lerp(a.r, b.r, t));
	const g = Math.round(lerp(a.g, b.g, t));
	const bl = Math.round(lerp(a.b, b.b, t));
	return '#'+_componentToHex(r)+_componentToHex(g)+_componentToHex(bl);
}

function drawRoundRect(ctx, x, y, w, h, r=6, fill=true, stroke=false){
	const radius = Math.min(r, w/2, h/2);
	ctx.beginPath();
	ctx.moveTo(x+radius, y);
	ctx.arcTo(x+w, y,   x+w, y+h, radius);
	ctx.arcTo(x+w, y+h, x,   y+h, radius);
	ctx.arcTo(x,   y+h, x,   y,   radius);
	ctx.arcTo(x,   y,   x+w, y,   radius);
	ctx.closePath();
	if (fill) ctx.fill();
	if (stroke) ctx.stroke();
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight){
	if (!text) return 0;
	const words = text.split(' ');
	let line = '', ty = y, lines = 0;
	for (let n=0; n<words.length; n++){
		const testLine = line + words[n] + ' ';
		const metrics = ctx.measureText(testLine);
		const testWidth = metrics.width;
		if (testWidth > maxWidth && n > 0){ ctx.fillText(line, x, ty); line = words[n] + ' '; ty += lineHeight; lines++; }
		else { line = testLine; }
	}
	if (line !== ''){ ctx.fillText(line, x, ty); lines++; }
	return lines;
}

function distanceTo(a, b){
	const dx = (a.x - b.x), dy = (a.y - b.y); return Math.hypot(dx, dy);
}

function tileToWorld(tx, ty){
	const ts = (typeof window !== 'undefined' && window.TILE_SIZE) ? window.TILE_SIZE : 16;
	return { x: tx * ts + ts/2, y: ty * ts + ts/2 };
}

function worldToTile(wx, wy){
	const ts = (typeof window !== 'undefined' && window.TILE_SIZE) ? window.TILE_SIZE : 16;
	return { tx: Math.floor(wx / ts), ty: Math.floor(wy / ts) };
}

// Exports (named) and attach to window for legacy/global usage
export { lerp, clamp, randInt, randFloat, hexToRgb, lerpColor, drawRoundRect, wrapText, distanceTo, tileToWorld, worldToTile };

if (typeof window !== 'undefined'){
	Object.assign(window, { lerp, clamp, randInt, randFloat, hexToRgb, lerpColor, drawRoundRect, wrapText, distanceTo, tileToWorld, worldToTile });
}
