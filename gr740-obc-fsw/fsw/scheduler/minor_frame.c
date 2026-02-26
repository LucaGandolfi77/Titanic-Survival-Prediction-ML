/**
 * @file minor_frame.c
 * @brief Minor/Major Frame Scheduler implementation.
 *
 * Executes registered slots at their configured rate using a simple
 * modulus-based phase check on the minor frame counter.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "minor_frame.h"

extern uint32_t bsp_get_uptime_ms(void);

/* ── Slot descriptor ───────────────────────────────────────────────────── */
typedef struct {
    char             name[16];
    sched_slot_fn_t  fn;
    uint32_t         rate;
    uint8_t          offset;
    uint8_t          priority;
    uint8_t          active;
} sched_slot_t;

/* ── Module state ──────────────────────────────────────────────────────── */
static sched_slot_t slots[SCHED_MAX_SLOTS];
static uint32_t     slot_count;
static uint32_t     minor_frame_counter;   /* global monotonic counter    */
static uint32_t     major_frame_counter;
static uint8_t      overrun_flag;
static uint8_t      sched_init_done = 0U;

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t sched_init(void)
{
    uint32_t i;

    for (i = 0U; i < SCHED_MAX_SLOTS; i++) {
        slots[i].fn       = (sched_slot_fn_t)0;
        slots[i].rate     = 0U;
        slots[i].offset   = 0U;
        slots[i].priority = 0U;
        slots[i].active   = 0U;
        slots[i].name[0]  = '\0';
    }
    slot_count          = 0U;
    minor_frame_counter = 0U;
    major_frame_counter = 0U;
    overrun_flag        = 0U;
    sched_init_done     = 1U;

    return SCHED_OK;
}

int32_t sched_register_slot(const char *name,
                             sched_slot_fn_t fn,
                             sched_rate_t rate,
                             uint8_t offset,
                             uint8_t priority)
{
    uint32_t i;

    if ((fn == (sched_slot_fn_t)0) || (rate == 0U)) {
        return SCHED_ERR_PARAM;
    }
    if (slot_count >= SCHED_MAX_SLOTS) {
        return SCHED_ERR_OVERFLOW;
    }

    {
        sched_slot_t *s = &slots[slot_count];
        s->fn       = fn;
        s->rate     = (uint32_t)rate;
        s->offset   = offset;
        s->priority = priority;
        s->active   = 1U;

        /* Copy name */
        if (name != (const char *)0) {
            for (i = 0U; (i < 15U) && (name[i] != '\0'); i++) {
                s->name[i] = name[i];
            }
            s->name[i] = '\0';
        } else {
            s->name[0] = '\0';
        }
    }

    slot_count++;
    return SCHED_OK;
}

int32_t sched_tick(void)
{
    uint32_t i;
    uint32_t phase;
    int32_t  executed = 0;
    uint32_t t_start;
    uint32_t t_end;

    if (sched_init_done == 0U) {
        return SCHED_ERR_PARAM;
    }

    t_start = bsp_get_uptime_ms();
    overrun_flag = 0U;

    phase = minor_frame_counter % SCHED_MAJOR_FRAME_CNT;

    /* Execute matching slots — priority sorted later if needed */
    for (i = 0U; i < slot_count; i++) {
        if (slots[i].active == 0U) {
            continue;
        }
        if (((minor_frame_counter - (uint32_t)slots[i].offset) %
              slots[i].rate) == 0U) {
            slots[i].fn();
            executed++;
        }
    }

    /* Advance minor frame */
    minor_frame_counter++;
    if (phase == (SCHED_MAJOR_FRAME_CNT - 1U)) {
        major_frame_counter++;
    }

    /* Check for overrun */
    t_end = bsp_get_uptime_ms();
    if ((t_end - t_start) >= SCHED_MINOR_FRAME_MS) {
        overrun_flag = 1U;
    }

    return executed;
}

uint32_t sched_get_minor_frame(void)
{
    return minor_frame_counter % SCHED_MAJOR_FRAME_CNT;
}

uint32_t sched_get_major_frame(void)
{
    return major_frame_counter;
}

uint8_t sched_overrun_detected(void)
{
    return overrun_flag;
}
