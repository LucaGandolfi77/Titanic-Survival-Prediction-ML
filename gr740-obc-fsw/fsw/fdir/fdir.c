/**
 * @file fdir.c
 * @brief FDIR — Fault Detection, Isolation, and Recovery implementation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "fdir.h"

extern uint32_t bsp_get_uptime_ms(void);

/* Forward declaration — mode manager safe-mode trigger */
extern int32_t mode_request_safe(void);

/* Forward declaration — event reporting */
extern int32_t pus_st05_raise_event(uint16_t event_id, uint8_t severity,
                                     const uint8_t *aux_data,
                                     uint16_t aux_len);

/* ── Module state ──────────────────────────────────────────────────────── */
static fdir_fault_t    fault_table[FDIR_MAX_FAULTS];
static uint32_t        fault_count = 0U;

static fdir_history_t  history[FDIR_MAX_HISTORY];
static uint32_t        history_head = 0U;
static uint32_t        history_used = 0U;

static uint8_t         fdir_init_done = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

static fdir_fault_t *find_fault(error_code_t err_code)
{
    uint32_t i;
    for (i = 0U; i < fault_count; i++) {
        if (fault_table[i].err_code == err_code) {
            return &fault_table[i];
        }
    }
    return (fdir_fault_t *)0;
}

static void record_history(error_code_t err_code, fdir_level_t level,
                            int32_t result)
{
    fdir_history_t *h = &history[history_head];
    h->err_code        = err_code;
    h->level           = level;
    h->timestamp_ms    = bsp_get_uptime_ms();
    h->recovery_result = result;

    history_head = (history_head + 1U) % FDIR_MAX_HISTORY;
    if (history_used < FDIR_MAX_HISTORY) {
        history_used++;
    }
}

static void escalate(fdir_fault_t *f, uint32_t aux)
{
    if (f->level == FDIR_LEVEL_1) {
        f->level = FDIR_LEVEL_2;
    } else if (f->level == FDIR_LEVEL_2) {
        f->level = FDIR_LEVEL_3;
    }

    if (f->level == FDIR_LEVEL_3) {
        /* Ultimate recovery: request safe mode */
        (void)mode_request_safe();
    }

    /* Raise event for escalation */
    {
        uint8_t ev_data[4];
        ev_data[0] = (uint8_t)(((uint16_t)f->err_code >> 8U) & 0xFFU);
        ev_data[1] = (uint8_t)((uint16_t)f->err_code & 0xFFU);
        ev_data[2] = (uint8_t)f->level;
        ev_data[3] = 0U;
        (void)pus_st05_raise_event((uint16_t)f->err_code,
                                    error_get_severity(f->err_code),
                                    ev_data, 4U);
    }

    record_history(f->err_code, f->level, -1);
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t fdir_init(void)
{
    uint32_t i;

    for (i = 0U; i < FDIR_MAX_FAULTS; i++) {
        fault_table[i].err_code         = ERR_NONE;
        fault_table[i].level            = FDIR_LEVEL_1;
        fault_table[i].recovery_fn      = (fdir_recovery_fn_t)0;
        fault_table[i].occurrence_count = 0U;
        fault_table[i].last_time_ms     = 0U;
        fault_table[i].enabled          = 0U;
        fault_table[i].active           = 0U;
    }
    fault_count = 0U;

    for (i = 0U; i < FDIR_MAX_HISTORY; i++) {
        history[i].err_code        = ERR_NONE;
        history[i].level           = FDIR_LEVEL_1;
        history[i].timestamp_ms    = 0U;
        history[i].recovery_result = 0;
    }
    history_head = 0U;
    history_used = 0U;

    fdir_init_done = 1U;
    return FDIR_OK;
}

int32_t fdir_register_fault(error_code_t err_code,
                             fdir_level_t level,
                             fdir_recovery_fn_t recovery)
{
    fdir_fault_t *existing;

    if (err_code == ERR_NONE) {
        return FDIR_ERR_PARAM;
    }

    existing = find_fault(err_code);
    if (existing != (fdir_fault_t *)0) {
        /* Update existing */
        existing->level       = level;
        existing->recovery_fn = recovery;
        existing->enabled     = 1U;
        return FDIR_OK;
    }

    if (fault_count >= FDIR_MAX_FAULTS) {
        return FDIR_ERR_FULL;
    }

    fault_table[fault_count].err_code         = err_code;
    fault_table[fault_count].level            = level;
    fault_table[fault_count].recovery_fn      = recovery;
    fault_table[fault_count].occurrence_count = 0U;
    fault_table[fault_count].last_time_ms     = 0U;
    fault_table[fault_count].enabled          = 1U;
    fault_table[fault_count].active           = 0U;
    fault_count++;

    return FDIR_OK;
}

int32_t fdir_report_fault(error_code_t err_code, uint32_t aux)
{
    fdir_fault_t *f;
    int32_t       result;

    if (fdir_init_done == 0U) {
        return FDIR_ERR_PARAM;
    }

    f = find_fault(err_code);
    if (f == (fdir_fault_t *)0) {
        return FDIR_ERR_PARAM;
    }
    if (f->enabled == 0U) {
        return FDIR_OK;
    }

    f->occurrence_count++;
    f->last_time_ms = bsp_get_uptime_ms();
    f->active = 1U;

    /* Attempt recovery */
    if (f->recovery_fn != (fdir_recovery_fn_t)0) {
        result = f->recovery_fn(err_code, aux);
        record_history(err_code, f->level, result);

        if (result == 0) {
            /* Recovery succeeded */
            f->active = 0U;
            return FDIR_OK;
        }
    } else {
        result = -1;
        record_history(err_code, f->level, result);
    }

    /* Check for escalation */
    if (f->occurrence_count >= FDIR_RETRY_LIMIT) {
        escalate(f, aux);
        f->occurrence_count = 0U;  /* Reset for next level */
    }

    return result;
}

int32_t fdir_clear_fault(error_code_t err_code)
{
    fdir_fault_t *f = find_fault(err_code);

    if (f == (fdir_fault_t *)0) {
        return FDIR_ERR_PARAM;
    }

    f->active           = 0U;
    f->occurrence_count = 0U;
    return FDIR_OK;
}

int32_t fdir_is_active(error_code_t err_code)
{
    fdir_fault_t *f = find_fault(err_code);

    if (f == (fdir_fault_t *)0) {
        return FDIR_ERR_PARAM;
    }

    return (f->active != 0U) ? 1 : 0;
}

uint32_t fdir_active_count(void)
{
    uint32_t i;
    uint32_t count = 0U;

    for (i = 0U; i < fault_count; i++) {
        if (fault_table[i].active != 0U) {
            count++;
        }
    }
    return count;
}

int32_t fdir_get_history(fdir_history_t *hist, uint32_t max_entries)
{
    uint32_t i;
    uint32_t idx;
    uint32_t to_copy;

    if (hist == (fdir_history_t *)0) {
        return FDIR_ERR_PARAM;
    }

    to_copy = (max_entries < history_used) ? max_entries : history_used;

    /* Read in reverse chronological order */
    for (i = 0U; i < to_copy; i++) {
        idx = (history_head + FDIR_MAX_HISTORY - 1U - i) % FDIR_MAX_HISTORY;
        hist[i] = history[idx];
    }

    return (int32_t)to_copy;
}

int32_t fdir_tick(void)
{
    /* Periodic check for persistent faults — placeholder for
     * more sophisticated detection logic (e.g. consecutive-frame
     * counters, sliding-window analysis, etc.)                  */
    (void)fdir_init_done;
    return 0;
}
