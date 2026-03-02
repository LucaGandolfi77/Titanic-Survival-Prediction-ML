// HUD updates (Speedometer, minimap, top-left notifications)
import { formatTime } from './utils.js';
import { CELL_THEMES } from './hypercube.js';

export class HUD {
    constructor() {
        this.speedValue = document.getElementById('speed-value');
        this.speedPath = document.getElementById('speed-value-path');
        this.scoreValue = document.getElementById('score-value');
        this.levelLabel = document.getElementById('level-label');
        this.deliveriesList = document.getElementById('hud-deliveries');
        
        this.minimapCanvas = document.getElementById('minimap-canvas');
        if (this.minimapCanvas) {
            this.minimapCtx = this.minimapCanvas.getContext('2d');
        }
        this.cellNameLabel = document.getElementById('cell-name-label');
    }
    
    updateSpeed(speed, maxSpeed) {
        if (!this.speedValue) return;
        const absSpeed = Math.abs(speed);
        // speed format
        this.speedValue.textContent = Math.floor(absSpeed * 3); // arbitrary multiplier for visual
        
        if (this.speedPath) {
            const ratio = absSpeed / maxSpeed;
            // total dash length is ~125
            const offset = 125 - (ratio * 125);
            this.speedPath.style.strokeDashoffset = offset;
            
            // color mapping
            if (ratio > 0.8) {
                this.speedPath.style.stroke = 'var(--accent-red)';
            } else if (ratio > 0.5) {
                this.speedPath.style.stroke = 'var(--accent-amber)';
            } else {
                this.speedPath.style.stroke = 'var(--accent-green)';
            }
        }
    }
    
    updateScore(score, level, targetDeliveries) {
        if (this.scoreValue) this.scoreValue.textContent = score;
        if (this.levelLabel) this.levelLabel.textContent = `LEVEL ${level} | GOAL: ${targetDeliveries}`;
    }
    
    updatePackages(activePackages) {
        if (!this.deliveriesList) return;
        this.deliveriesList.innerHTML = '';
        
        activePackages.forEach(pkg => {
            if (pkg.state === 'carried') {
                const el = document.createElement('div');
                const isUrgent = pkg.timeRemaining < 15;
                el.className = `delivery-card ${isUrgent ? 'urgent' : ''}`;
                
                const timeRatio = pkg.timeRemaining / pkg.timeLimit;
                const widthPct = Math.max(0, timeRatio * 100);
                
                el.innerHTML = `
                    <div style="display:flex; justify-content:space-between">
                        <span>📦 To Cell ${pkg.destinationCell}</span>
                        <span>⏱ ${formatTime(pkg.timeRemaining)}</span>
                    </div>
                    <div class="time-bar" style="width: ${widthPct}%"></div>
                `;
                this.deliveriesList.appendChild(el);
            }
        });
    }

    updateMinimap(currentCell, activePackages) {
        if (!this.minimapCtx) return;
        
        if (this.cellNameLabel) {
            this.cellNameLabel.textContent = `CELL ${currentCell} - ${CELL_THEMES[currentCell].name.toUpperCase()}`;
            this.cellNameLabel.style.color = CELL_THEMES[currentCell].accent;
        }

        const ctx = this.minimapCtx;
        ctx.clearRect(0, 0, 140, 140);
        
        // Very basic 8-node drawing (could be improved to actual 2D projection of hypercube)
        // Nodes: inner 4, outer 4 layout
        const nodes = [
            {id:0, x:50, y:50}, {id:1, x:90, y:50}, {id:2, x:50, y:90}, {id:3, x:90, y:90},
            {id:4, x:30, y:30}, {id:5, x:110, y:30}, {id:6, x:30, y:110}, {id:7, x:110, y:110}
        ];
        
        // Draw edges (simplified)
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        
        // just draw links between 0,1,2,3 square and 4,5,6,7 square for aesthetic
        const drawLine = (p1, p2) => {
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        };
        drawLine(nodes[0], nodes[1]); drawLine(nodes[1], nodes[3]);
        drawLine(nodes[3], nodes[2]); drawLine(nodes[2], nodes[0]);
        drawLine(nodes[4], nodes[5]); drawLine(nodes[5], nodes[7]);
        drawLine(nodes[7], nodes[6]); drawLine(nodes[6], nodes[4]);
        for(let i=0; i<4; i++) drawLine(nodes[i], nodes[i+4]);
        
        // Draw nodes
        nodes.forEach(n => {
            // is it destination?
            const isDest = activePackages.some(p => p.state === 'carried' && p.destinationCell === n.id);
            
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.id === currentCell ? 6 : 4, 0, Math.PI*2);
            
            if (n.id === currentCell) {
                ctx.fillStyle = CELL_THEMES[n.id].accent;
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = 10;
            } else if (isDest) {
                ctx.fillStyle = '#ff1744';
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = 10;
            } else {
                ctx.fillStyle = CELL_THEMES[n.id].accent;
                ctx.shadowBlur = 0;
            }
            
            ctx.fill();
        });
        
        ctx.shadowBlur = 0; // reset
    }
}