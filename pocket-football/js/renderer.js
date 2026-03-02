export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false }); // Optimize
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.ratio = window.devicePixelRatio || 1;
    
    // Virtual Field (16:9)
    this.fieldW = 800;
    this.fieldH = 500; // 16:10 roughly, more mobile-ish 
    // Wait, requirement says 800x500 is 1.6 aspect ratio. 16:10.
    
    // Scale factor
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }
  
  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width * this.ratio;
    this.canvas.height = this.height * this.ratio;
    this.ctx.scale(this.ratio, this.ratio);
    
    // Calculate fit
    const aspect = this.width / this.height;
    const fieldAspect = this.fieldW / this.fieldH;
    
    if (aspect > fieldAspect) {
      // Screen is wider than field -> fit to height
      this.scale = this.height / this.fieldH;
      this.offsetX = (this.width - this.fieldW * this.scale) / 2;
      this.offsetY = 0;
    } else {
      // Screen is taller/narrower -> fit to width
      this.scale = this.width / this.fieldW;
      this.offsetX = 0;
      this.offsetY = (this.height - this.fieldH * this.scale) / 2;
    }
    
    // Update CSS for controls to match field if needed, but UI is overlaid full screen
  }
  
  render(state) {
    const { players, ball, matchTime, scores, isGoal, controlledPlayer } = state;
    
    // Clear
    this.ctx.fillStyle = '#1a1a2e'; // Dark background outside field
    this.ctx.fillRect(0, 0, this.width, this.height);
    
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);
    
    // 1. Field
    this.drawField();
    
    // 2. Shadows (Players + Ball)
    this.ctx.fillStyle = 'rgba(0,0,0,0.2)';
    players.forEach(p => {
      this.ctx.beginPath();
      this.ctx.ellipse(p.pos.x, p.pos.y + 10, 8, 4, 0, 0, Math.PI * 2);
      this.ctx.fill();
    });
    // Ball Shadow
    if (ball) {
      this.ctx.beginPath();
      const shadowScale = 1 - (ball.height / 50); // Shrink with height
      this.ctx.ellipse(ball.pos.x, ball.pos.y + 4 + ball.height * 0.2, 
                       6 * shadowScale, 3 * shadowScale, 0, 0, Math.PI * 2);
      this.ctx.fill();
    }
    
    // 3. Goal flash
    if (isGoal) {
      this.ctx.fillStyle = 'rgba(253, 224, 71, 0.3)'; // Yellow flash
      // Determine side? Assuming full field flash or specific goal
      // Let's flash the scored goal area
      if (ball.pos.x < 400) {
        this.ctx.fillRect(-30, 210, 40, 80);
      } else {
        this.ctx.fillRect(790, 210, 40, 80);
      }
    }

    // 4. Players
    players.forEach(p => {
      this.drawPlayer(p, p === controlledPlayer);
    });
    
    // 5. Ball
    if (ball) this.drawBall(ball);
    
    this.ctx.restore();
    
    // Minimap (Top Right) -- Optional per spec, implementing basic version
    // this.drawMinimap(players, ball);
  }
  
  drawField() {
    // Grass Base
    this.ctx.fillStyle = '#4a8f3f';
    this.ctx.fillRect(0, 0, this.fieldW, this.fieldH);
    
    // Stripes
    this.ctx.fillStyle = '#3f7a35';
    const stripeW = this.fieldW / 10;
    for (let i = 0; i < 10; i++) {
       if (i % 2 === 1) {
         this.ctx.fillRect(i * stripeW, 0, stripeW, this.fieldH);
       }
    }
    
    // Lines
    this.ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    this.ctx.lineWidth = 3;
    this.ctx.beginPath();
    
    // Outer Border
    this.ctx.strokeRect(0, 0, this.fieldW, this.fieldH);
    
    // Center Line
    this.ctx.moveTo(this.fieldW / 2, 0);
    this.ctx.lineTo(this.fieldW / 2, this.fieldH);
    
    // Center Circle
    this.ctx.moveTo(this.fieldW / 2 + 60, this.fieldH / 2);
    this.ctx.arc(this.fieldW / 2, this.fieldH / 2, 60, 0, Math.PI * 2);
    
    // Center Spot
    this.ctx.moveTo(this.fieldW / 2, this.fieldH / 2); // Avoid subpath
    this.ctx.arc(this.fieldW / 2, this.fieldH / 2, 2, 0, Math.PI * 2);
    
    // Penalty Areas (Left)
    this.ctx.strokeRect(0, 110, 110, 280); // Height 280, Centered Y=250 -> 110-390
    this.ctx.strokeRect(0, 190, 40, 120);  // Goal Area: Height 120, Y=250 -> 190-310
    
    // Penalty Areas (Right)
    this.ctx.strokeRect(this.fieldW - 110, 110, 110, 280);
    this.ctx.strokeRect(this.fieldW - 40, 190, 40, 120);
    
    // Corners
    this.ctx.moveTo(15, 0); this.ctx.arc(0, 0, 15, 0, Math.PI * 0.5);
    this.ctx.moveTo(15, this.fieldH); this.ctx.arc(0, this.fieldH, 15, Math.PI * 1.5, 0); // Correct Arc
    // ... Simplified corners
    
    this.ctx.stroke();
    
    // Goals (Outside field)
    this.ctx.fillStyle = 'rgba(255,255,255,0.2)';
    // Left Goal: x=-30 to 0, y=210 to 290 (80px height)
    this.ctx.fillRect(-30, 210, 30, 80);
    this.ctx.strokeRect(-30, 210, 30, 80);
    
    // Right Goal
    this.ctx.fillRect(this.fieldW, 210, 30, 80);
    this.ctx.strokeRect(this.fieldW, 210, 30, 80);
  }
  
  drawPlayer(p, isControlled) {
    const x = p.pos.x;
    const y = p.pos.y;
    
    // Selection Ring
    if (isControlled) {
      // Pulsing
      const pulse = (Math.sin(performance.now() * 0.01) + 1) * 0.5; // 0-1
      this.ctx.strokeStyle = `rgba(253, 224, 71, ${0.5 + pulse * 0.5})`; // Yellow
      this.ctx.lineWidth = 3;
      this.ctx.beginPath();
      this.ctx.arc(x, y, 18 + pulse * 2, 0, Math.PI * 2);
      this.ctx.stroke();
      
      // Control Arrow
      this.ctx.fillStyle = '#fde047';
      this.ctx.beginPath();
      this.ctx.moveTo(x - 5, y - 28 - pulse * 2);
      this.ctx.lineTo(x + 5, y - 28 - pulse * 2);
      this.ctx.lineTo(x, y - 20 - pulse * 2);
      this.ctx.fill();
      
      // Stamina Bar
      if (p.stamina < 100) {
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(x - 10, y + 20, 20, 4);
        this.ctx.fillStyle = p.stamina < 30 ? '#ef4444' : '#22c55e';
        this.ctx.fillRect(x - 10, y + 20, 20 * (p.stamina / 100), 4);
      }
    }

    // Body
    this.ctx.fillStyle = p.team === 0 ? '#3b82f6' : '#ef4444'; // Blue vs Red
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(x, y, 14, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.stroke();
    
    // Number
    this.ctx.fillStyle = 'white';
    this.ctx.font = 'bold 12px Inter';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(p.number, x, y);
    
    // Direction Indicator
    this.ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
    this.ctx.lineTo(x + Math.cos(p.facing) * 14, y + Math.sin(p.facing) * 14);
    this.ctx.stroke();
  }
  
  drawBall(b) {
    const x = b.pos.x;
    const y = b.pos.y - b.height; // Visual offset for height (simple 2D fake)
    
    // Ball Body
    this.ctx.fillStyle = '#f8fafc';
    this.ctx.beginPath();
    this.ctx.arc(x, y, 8, 0, Math.PI * 2);
    this.ctx.fill();
    
    // Detail (Pentagons approx)
    this.ctx.fillStyle = '#1e293b'; // Dark blue-black
    // Simple rotation based on velocity or spin?
    // Just drawing fixed pattern for now as rotation in 2d is subtle at this scale
    this.ctx.beginPath();
    this.ctx.arc(x, y, 4, 0, Math.PI * 2);
    this.ctx.fill(); // Center dot
  }
}