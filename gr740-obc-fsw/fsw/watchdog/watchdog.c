/**
 * @file watchdog.c
 * @brief Watchdog Manager implementation.
 *
 * HW watchdog uses GPTIMER channel for countdown reset.
 * SW watchdog is a heartbeat table tracked via uptime comparisons.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "watchdog.h"
#include "../../config/hw_config.h"

extern uint32_t bsp_get_uptime_ms(void);

/* ── GR740 watchdog register offsets (GPTIMER unit 0, channel 3) ───── */
/* Reusing GPTIMER registers; WDT is configured as a one-shot
 * countdown timer that resets the system on underflow.               */
#define WDG_TIMER_BASE   GPTIMER0_BASE_ADDR
#define WDG_CHANNEL      3U
#define WDG_CH_OFFSET    (0x10U + (WDG_CHANNEL * 0x10U))

#define WDG_CH_COUNTER   (WDG_TIMER_BASE + WDG_CH_OFFSET + 0x00U)
#define WDG_CH_RELOAD    (WDG_TIMER_BASE + WDG_CH_OFFSET + 0x04U)
#define WDG_CH_CTRL      (WDG_TIMER_BASE + WDG_CH_OFFSET + 0x08U)

/* Timer control bits */
#define WDG_CTRL_EN      (1U << 0U)   /* Enable               */
#define WDG_CTRL_RS      (1U << 1U)   /* Restart (reload)     */
#define WDG_CTRL_LD      (1U << 2U)   /* Load counter         */
#define WDG_CTRL_IE      (1U << 3U)   /* Interrupt enable     */

/* ── Software heartbeat entry ──────────────────────────────────────── */
typedef struct {
    uint32_t task_id;
    uint32_t timeout_ms;
    uint32_t last_beat_ms;
    char     name[16];
    uint8_t  active;
} heartbeat_entry_t;

/* ── Module state ──────────────────────────────────────────────────── */
static heartbeat_entry_t hb_table[WDG_MAX_MONITORED];
static uint32_t          hb_count = 0U;
static uint8_t           hw_enabled = 0U;
static uint8_t           wdg_init_done = 0U;

/* tick count for 2s at 1 µs per tick = 2,000,000 */
#define WDG_HW_RELOAD_VAL  (WDG_HW_TIMEOUT_MS * 1000U)

/* ── Public API ────────────────────────────────────────────────────── */

int32_t wdg_init(void)
{
    uint32_t i;

    for (i = 0U; i < WDG_MAX_MONITORED; i++) {
        hb_table[i].task_id     = 0U;
        hb_table[i].timeout_ms  = 0U;
        hb_table[i].last_beat_ms = 0U;
        hb_table[i].active      = 0U;
        hb_table[i].name[0]     = '\0';
    }
    hb_count     = 0U;
    hw_enabled   = 0U;
    wdg_init_done = 1U;

    /* Configure HW watchdog timer — load but do NOT enable yet */
    REG_WRITE(WDG_CH_RELOAD, WDG_HW_RELOAD_VAL);
    REG_WRITE(WDG_CH_CTRL, WDG_CTRL_LD);  /* Load counter from reload */

    return WDG_OK;
}

void wdg_hw_kick(void)
{
    if (hw_enabled != 0U) {
        /* Reload the counter to prevent underflow reset */
        REG_WRITE(WDG_CH_CTRL, WDG_CTRL_EN | WDG_CTRL_RS | WDG_CTRL_LD);
    }
}

void wdg_hw_enable(void)
{
    hw_enabled = 1U;
    REG_WRITE(WDG_CH_RELOAD, WDG_HW_RELOAD_VAL);
    REG_WRITE(WDG_CH_CTRL, WDG_CTRL_EN | WDG_CTRL_RS | WDG_CTRL_LD | WDG_CTRL_IE);
}

void wdg_hw_disable(void)
{
    hw_enabled = 0U;
    REG_WRITE(WDG_CH_CTRL, 0U);
}

int32_t wdg_register_task(uint32_t task_id, uint32_t timeout_ms,
                           const char *name)
{
    uint32_t i;
    heartbeat_entry_t *e;

    if (hb_count >= WDG_MAX_MONITORED) {
        return WDG_ERR_PARAM;
    }
    if (timeout_ms == 0U) {
        return WDG_ERR_PARAM;
    }

    e = &hb_table[hb_count];
    e->task_id     = task_id;
    e->timeout_ms  = timeout_ms;
    e->last_beat_ms = bsp_get_uptime_ms();
    e->active       = 1U;

    if (name != (const char *)0) {
        for (i = 0U; (i < 15U) && (name[i] != '\0'); i++) {
            e->name[i] = name[i];
        }
        e->name[i] = '\0';
    } else {
        e->name[0] = '\0';
    }

    hb_count++;
    return WDG_OK;
}

int32_t wdg_heartbeat(uint32_t task_id)
{
    uint32_t i;

    for (i = 0U; i < hb_count; i++) {
        if ((hb_table[i].active != 0U) &&
            (hb_table[i].task_id == task_id)) {
            hb_table[i].last_beat_ms = bsp_get_uptime_ms();
            return WDG_OK;
        }
    }
    return WDG_ERR_PARAM;
}

int32_t wdg_check_all(uint32_t *expired_id)
{
    uint32_t i;
    uint32_t now;
    int32_t  timed_out = 0;
    uint8_t  first_found = 0U;

    if (wdg_init_done == 0U) {
        return 0;
    }

    now = bsp_get_uptime_ms();

    for (i = 0U; i < hb_count; i++) {
        if (hb_table[i].active == 0U) {
            continue;
        }
        if ((now - hb_table[i].last_beat_ms) > hb_table[i].timeout_ms) {
            timed_out++;
            if ((first_found == 0U) && (expired_id != (uint32_t *)0)) {
                *expired_id = hb_table[i].task_id;
                first_found = 1U;
            }
        }
    }

    return timed_out;
}
