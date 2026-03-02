import { domUtils } from './utils.js';

export class HUD {
    constructor() {
        this.scoreEl = domUtils.get('score-display');
        this.spillsEl = domUtils.get('spills-display');
        this.gravityArrow = domUtils.get('gravity-arrow-container');
        this.gravityStateName = domUtils.get('gravity-state-name');
        this.gravityTimerFill = domUtils.get('gravity-timer-fill');
        this.warningEl = domUtils.get('gravity-warning');
        this.crosshair = document.querySelector('.ch-dot');
        
        this.heldInfo = domUtils.get('hud-held-info');
        this.heldName = domUtils.get('held-name');
        this.tiltArc = domUtils.get('tilt-arc');
        this.heldFill = domUtils.get('held-pour-fill');
        
        this.ordersContainer = domUtils.get('hud-orders');
        this.orders = {}; // DOM map
        
        this.pourMeter = domUtils.get('hud-pour-meter');
        this.pourCurrent = domUtils.get('pour-fill-current');
        this.pourTarget = domUtils.get('pour-fill-target');
        
        this.hudWrapper = domUtils.get('hud-wrapper');
    }
    
    updateScore(score) {
        this.scoreEl.innerText = `☕ ${score}`;
    }
    
    updateSpills(spills) {
        this.spillsEl.innerText = `💧 ${spills}`;
    }
    
    updateGravityDisplay(vector, name, timeRemaining, maxTime) {
        this.gravityStateName.innerText = name;
        
        // Simplistic vector to arrow rotation (assuming looking top-down on X/Y plane for HUD)
        let angle = Math.atan2(vector.x, -vector.y) * 180 / Math.PI;
        if(vector.z < -5) angle = 180; // Forward
        if(vector.z > 5) angle = 0;   // Backward
        if(vector.length() < 1) angle = 0; // ZeroG
        
        this.gravityArrow.style.transform = `rotate(${angle}deg)`;
        
        const pct = Math.max(0, (timeRemaining / maxTime)) * 100;
        this.gravityTimerFill.style.width = `${pct}%`;
        
        if (timeRemaining <= 3) {
            this.gravityTimerFill.style.background = 'var(--accent-red)';
        } else {
            this.gravityTimerFill.style.background = 'var(--accent-cyan)';
        }
        
        // CSS shift shake (visual only)
        if( timeRemaining <= 0.2 || (timeRemaining > maxTime - 1.0)) {
           this.hudWrapper.classList.add('shake-active');
        } else {
           this.hudWrapper.classList.remove('shake-active');
        }
    }
    
    showWarning(show) {
        if(show) {
            this.warningEl.classList.remove('hidden');
            document.body.classList.add('warning-state');
        } else {
            this.warningEl.classList.add('hidden');
            document.body.classList.remove('warning-state');
        }
    }
    
    setCrosshairHover(isActive) {
        if(isActive) this.crosshair.classList.add('ch-active');
        else this.crosshair.classList.remove('ch-active');
    }
    
    showHeldInfo(show) {
        if(show) this.heldInfo.classList.remove('hidden');
        else this.heldInfo.classList.add('hidden');
    }
    
    updateHeldInfo(name, tiltAngle, fillLevel) {
        this.heldName.innerText = name;
        
        // Update tilt indicator pseudo-element position via inline style variable hack or math
        const pct = (tiltAngle / Math.PI) * 100;
        this.tiltArc.style.background = `linear-gradient(90deg, var(--accent-cyan) ${pct}%, #333 ${pct}%)`;
        
        if (fillLevel !== null) {
            this.heldFill.style.width = `${fillLevel * 100}%`;
            this.heldFill.style.display = 'block';
        } else {
            this.heldFill.style.display = 'none';
        }
    }
    
    showPourMeter(show, currentFill = 0, targetFill = 0.8) {
        if(show) {
            this.pourMeter.classList.remove('hidden');
            this.pourCurrent.style.height = `${currentFill * 100}%`;
            
            // Set target zone limits (approx +- 10%)
            const minT = Math.max(0, targetFill - 0.1);
            const h = 0.2; // 20% tolerance block
            this.pourTarget.style.bottom = `${minT * 100}%`;
            this.pourTarget.style.height = `${h * 100}%`;
        } else {
            this.pourMeter.classList.add('hidden');
        }
    }
    
    addOrder(order) {
        const div = document.createElement('div');
        div.className = 'order-card';
        div.id = `order-${order.id}`;
        
        div.innerHTML = `
            <div class="order-avatar">${order.icon}</div>
            <div class="order-details">
                <div class="order-target-fill">
                    <div class="order-target-zone" style="left:${(order.targetFill-0.1)*100}%; width:20%;"></div>
                </div>
                <div class="order-timer"><div class="order-timer-fill" id="timer-${order.id}"></div></div>
            </div>
        `;
        
        this.ordersContainer.appendChild(div);
        this.orders[order.id] = div;
    }
    
    updateOrder(order) {
        const timerEl = document.getElementById(`timer-${order.id}`);
        if(timerEl) {
            const pct = Math.max(0, order.timeLeft / order.maxTime) * 100;
            timerEl.style.width = `${pct}%`;
            if(pct < 25) timerEl.style.background = 'red';
            else if(pct < 50) timerEl.style.background = 'yellow';
        }
    }
    
    removeOrder(id) {
        const div = this.orders[id];
        if(div) {
            div.remove();
            delete this.orders[id];
        }
    }
}