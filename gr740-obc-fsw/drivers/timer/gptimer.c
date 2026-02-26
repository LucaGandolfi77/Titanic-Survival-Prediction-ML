/**
 * @file gptimer.c
 * @brief GPTIMER driver implementation for GR740.
 *
 * Register-level driver for the Cobham Gaisler GPTIMER core.
 * Each GPTIMER unit has a shared prescaler and up to 4 timer channels.
 * Supports periodic and one-shot modes with interrupt callbacks.
 *
 * Timer unit 0, channel 0 is reserved for BSP systick (1 ms).
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "gptimer.h"
#include "../../config/hw_config.h"

/* ── GPTIMER Register Offsets ──────────────────────────────────────────── */
#define GPTIMER_SCALER      0x00U   /**< Scaler value register        */
#define GPTIMER_SRELOAD     0x04U   /**< Scaler reload value          */
#define GPTIMER_CONFIG      0x08U   /**< Configuration register       */
#define GPTIMER_LATCH_CFG   0x0CU   /**< Latch configuration          */

/* Per-timer registers (base + 0x10 + n*0x10) */
#define GPTIMER_T_COUNTER(n)  (0x10U + ((n) * 0x10U) + 0x00U) /**< Counter value   */
#define GPTIMER_T_RELOAD(n)   (0x10U + ((n) * 0x10U) + 0x04U) /**< Reload value    */
#define GPTIMER_T_CTRL(n)     (0x10U + ((n) * 0x10U) + 0x08U) /**< Control register */
#define GPTIMER_T_LATCH(n)    (0x10U + ((n) * 0x10U) + 0x0CU) /**< Latch value     */

/* ── Configuration Register (read-only, capability) ────────────────────── */
#define CONFIG_NTIMERS_SHIFT    0U
#define CONFIG_NTIMERS_MASK     0x07U   /**< Number of timers (0=1, ...) */
#define CONFIG_IRQ_SHIFT        3U
#define CONFIG_IRQ_MASK         0xF8U   /**< IRQ number field            */
#define CONFIG_SI               (1U << 8)  /**< Separate interrupts      */
#define CONFIG_DF               (1U << 9)  /**< Disable freeze           */

/* ── Timer Control Register Bits ───────────────────────────────────────── */
#define TCTRL_EN        (1U << 0)   /**< Enable timer              */
#define TCTRL_RS        (1U << 1)   /**< Restart (auto-reload)     */
#define TCTRL_LD        (1U << 2)   /**< Load counter from reload  */
#define TCTRL_IE        (1U << 3)   /**< Interrupt enable          */
#define TCTRL_IP        (1U << 4)   /**< Interrupt pending (W1C)   */
#define TCTRL_CH        (1U << 5)   /**< Chain with preceding timer */
#define TCTRL_DH        (1U << 6)   /**< Debug halt                */

/* ── Module state ──────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t *base;
    uint8_t            initialized;
    uint8_t            num_timers;
    timer_callback_t   callbacks[TIMER_MAX_CHANNELS];
    uint8_t            oneshot[TIMER_MAX_CHANNELS];
} timer_unit_t;

static timer_unit_t timer_units[TIMER_MAX_UNITS];

/* ── Register access helpers ───────────────────────────────────────────── */
static inline uint32_t timer_reg_read(const timer_unit_t *unit, uint32_t offset)
{
    return *(volatile uint32_t *)((uint32_t)unit->base + offset);
}

static inline void timer_reg_write(timer_unit_t *unit, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uint32_t)unit->base + offset) = value;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t timer_init(uint32_t unit, uint32_t base_addr)
{
    uint32_t config_reg;
    uint32_t i;
    timer_unit_t *tu;

    if (unit >= TIMER_MAX_UNITS) {
        return TIMER_ERR_PARAM;
    }
    if (base_addr == 0U) {
        return TIMER_ERR_PARAM;
    }

    tu = &timer_units[unit];
    tu->base = (volatile uint32_t *)(uintptr_t)base_addr;

    /* Read capability register to determine number of timers */
    config_reg = timer_reg_read(tu, GPTIMER_CONFIG);
    tu->num_timers = (uint8_t)((config_reg & CONFIG_NTIMERS_MASK) + 1U);
    if (tu->num_timers > TIMER_MAX_CHANNELS) {
        tu->num_timers = TIMER_MAX_CHANNELS;
    }

    /* Initialize callbacks */
    for (i = 0U; i < TIMER_MAX_CHANNELS; i++) {
        tu->callbacks[i] = (timer_callback_t)0;
        tu->oneshot[i] = 0U;
    }

    /*
     * Configure prescaler for 1 MHz tick (1 µs resolution)
     * from 50 MHz system clock.
     * Scaler reload = (50 MHz / 1 MHz) - 1 = 49
     *
     * Note: If unit 0 is already configured by BSP for systick,
     * we preserve the scaler but still set it for consistency.
     */
    timer_reg_write(tu, GPTIMER_SRELOAD, 49U);
    timer_reg_write(tu, GPTIMER_SCALER, 49U);

    /* Stop all timers and clear pending interrupts */
    for (i = 0U; i < tu->num_timers; i++) {
        /* Skip channel 0 of unit 0 if BSP systick is running */
        if ((unit == 0U) && (i == 0U)) {
            continue;
        }
        timer_reg_write(tu, GPTIMER_T_CTRL(i), TCTRL_IP); /* Clear IP, disable */
    }

    tu->initialized = 1U;
    return TIMER_OK;
}

int32_t timer_configure(uint32_t unit, uint32_t channel,
                         uint32_t period_us, timer_callback_t cb)
{
    timer_unit_t *tu;

    if (unit >= TIMER_MAX_UNITS) {
        return TIMER_ERR_PARAM;
    }

    tu = &timer_units[unit];
    if (tu->initialized == 0U) {
        return TIMER_ERR_INIT;
    }
    if (channel >= tu->num_timers) {
        return TIMER_ERR_PARAM;
    }
    if (period_us == 0U) {
        return TIMER_ERR_PARAM;
    }

    /* Protect BSP systick */
    if ((unit == 0U) && (channel == 0U)) {
        return TIMER_ERR_BUSY;
    }

    /* Stop timer */
    timer_reg_write(tu, GPTIMER_T_CTRL(channel), TCTRL_IP);

    /*
     * With 1 MHz prescaler tick, reload = period_us - 1
     * Timer counts down from reload to 0, underflow triggers interrupt.
     */
    timer_reg_write(tu, GPTIMER_T_RELOAD(channel), period_us - 1U);

    /* Load counter from reload value */
    timer_reg_write(tu, GPTIMER_T_CTRL(channel), TCTRL_LD);

    tu->callbacks[channel] = cb;
    tu->oneshot[channel] = 0U; /* Periodic by default */

    return TIMER_OK;
}

int32_t timer_start(uint32_t unit, uint32_t channel)
{
    timer_unit_t *tu;
    uint32_t ctrl;

    if (unit >= TIMER_MAX_UNITS) {
        return TIMER_ERR_PARAM;
    }

    tu = &timer_units[unit];
    if (tu->initialized == 0U) {
        return TIMER_ERR_INIT;
    }
    if (channel >= tu->num_timers) {
        return TIMER_ERR_PARAM;
    }

    ctrl = TCTRL_EN | TCTRL_IE | TCTRL_LD;

    if (tu->oneshot[channel] == 0U) {
        ctrl |= TCTRL_RS; /* Auto-reload for periodic mode */
    }

    timer_reg_write(tu, GPTIMER_T_CTRL(channel), ctrl);

    return TIMER_OK;
}

int32_t timer_stop(uint32_t unit, uint32_t channel)
{
    timer_unit_t *tu;

    if (unit >= TIMER_MAX_UNITS) {
        return TIMER_ERR_PARAM;
    }

    tu = &timer_units[unit];
    if (tu->initialized == 0U) {
        return TIMER_ERR_INIT;
    }
    if (channel >= tu->num_timers) {
        return TIMER_ERR_PARAM;
    }

    /* Disable and clear pending */
    timer_reg_write(tu, GPTIMER_T_CTRL(channel), TCTRL_IP);

    return TIMER_OK;
}

int32_t timer_read_counter(uint32_t unit, uint32_t channel, uint32_t *value)
{
    timer_unit_t *tu;

    if (unit >= TIMER_MAX_UNITS) {
        return TIMER_ERR_PARAM;
    }

    tu = &timer_units[unit];
    if (tu->initialized == 0U) {
        return TIMER_ERR_INIT;
    }
    if (channel >= tu->num_timers) {
        return TIMER_ERR_PARAM;
    }
    if (value == (uint32_t *)0) {
        return TIMER_ERR_PARAM;
    }

    *value = timer_reg_read(tu, GPTIMER_T_COUNTER(channel));

    return TIMER_OK;
}

int32_t timer_oneshot(uint32_t unit, uint32_t channel,
                       uint32_t timeout_us, timer_callback_t cb)
{
    int32_t ret;
    timer_unit_t *tu;

    ret = timer_configure(unit, channel, timeout_us, cb);
    if (ret != TIMER_OK) {
        return ret;
    }

    tu = &timer_units[unit];
    tu->oneshot[channel] = 1U; /* One-shot mode */

    return timer_start(unit, channel);
}

void timer_isr(uint32_t unit)
{
    timer_unit_t *tu;
    uint32_t i;
    uint32_t ctrl;

    if (unit >= TIMER_MAX_UNITS) {
        return;
    }

    tu = &timer_units[unit];
    if (tu->initialized == 0U) {
        return;
    }

    for (i = 0U; i < tu->num_timers; i++) {
        ctrl = timer_reg_read(tu, GPTIMER_T_CTRL(i));

        if ((ctrl & TCTRL_IP) != 0U) {
            /* Clear interrupt pending (write-1-to-clear on IP bit) */
            timer_reg_write(tu, GPTIMER_T_CTRL(i), ctrl | TCTRL_IP);

            /* For one-shot, stop the timer */
            if (tu->oneshot[i] != 0U) {
                timer_reg_write(tu, GPTIMER_T_CTRL(i), TCTRL_IP);
            }

            /* Invoke callback */
            if (tu->callbacks[i] != (timer_callback_t)0) {
                tu->callbacks[i](unit, i);
            }
        }
    }
}

uint32_t timer_get_timestamp(void)
{
    const timer_unit_t *tu = &timer_units[1];

    if (tu->initialized == 0U) {
        return 0U;
    }

    return timer_reg_read(tu, GPTIMER_T_COUNTER(3));
}
