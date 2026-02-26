/**
 * @file mode_manager.c
 * @brief Mode Manager — operational mode FSM implementation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "mode_manager.h"

extern uint32_t bsp_get_uptime_ms(void);

/* ── Transition table ──────────────────────────────────────────────────── */
/*
 * allowed[from][to] = 1 means transition is permitted.
 * Transition to SAFE is always allowed (handled separately).
 */
static const uint8_t allowed[MODE_COUNT][MODE_COUNT] = {
    /*               BOOT  SAFE  NOM   SCI   ECL   DET  */
    /* BOOT     */ {  0,    1,    0,    0,    0,    0  },
    /* SAFE     */ {  0,    0,    1,    0,    0,    1  },
    /* NOMINAL  */ {  0,    1,    0,    1,    1,    0  },
    /* SCIENCE  */ {  0,    1,    1,    0,    0,    0  },
    /* ECLIPSE  */ {  0,    1,    1,    0,    0,    0  },
    /* DETUMBL  */ {  0,    1,    0,    0,    0,    0  }
};

/* ── Mode callbacks ────────────────────────────────────────────────────── */
typedef struct {
    mode_entry_fn_t on_entry;
    mode_exit_fn_t  on_exit;
} mode_callbacks_t;

/* ── Module state ──────────────────────────────────────────────────────── */
static obc_mode_t        current_mode;
static obc_mode_t        previous_mode;
static uint32_t          transition_count;
static mode_callbacks_t  callbacks[MODE_COUNT];
static uint8_t           mm_init_done = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

static int32_t do_transition(obc_mode_t target)
{
    int32_t ret;

    /* Exit callback for current mode */
    if (callbacks[current_mode].on_exit != (mode_exit_fn_t)0) {
        callbacks[current_mode].on_exit(current_mode, target);
    }

    /* Entry callback for target mode */
    if (callbacks[target].on_entry != (mode_entry_fn_t)0) {
        ret = callbacks[target].on_entry(current_mode, target);
        if (ret != 0) {
            /* Entry failed — stay in current mode */
            return MODE_ERR_ILLEGAL;
        }
    }

    previous_mode = current_mode;
    current_mode  = target;
    transition_count++;

    return MODE_OK;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t mode_manager_init(void)
{
    uint32_t i;

    current_mode     = MODE_BOOT;
    previous_mode    = MODE_BOOT;
    transition_count = 0U;

    for (i = 0U; i < (uint32_t)MODE_COUNT; i++) {
        callbacks[i].on_entry = (mode_entry_fn_t)0;
        callbacks[i].on_exit  = (mode_exit_fn_t)0;
    }

    mm_init_done = 1U;
    return MODE_OK;
}

int32_t mode_register_callbacks(obc_mode_t mode,
                                 mode_entry_fn_t on_entry,
                                 mode_exit_fn_t on_exit)
{
    if ((uint32_t)mode >= (uint32_t)MODE_COUNT) {
        return MODE_ERR_PARAM;
    }

    callbacks[mode].on_entry = on_entry;
    callbacks[mode].on_exit  = on_exit;
    return MODE_OK;
}

int32_t mode_request_transition(obc_mode_t target)
{
    if (mm_init_done == 0U) {
        return MODE_ERR_PARAM;
    }
    if ((uint32_t)target >= (uint32_t)MODE_COUNT) {
        return MODE_ERR_PARAM;
    }
    if (target == current_mode) {
        return MODE_OK;  /* Already in target mode */
    }

    /* SAFE is always reachable */
    if (target == MODE_SAFE) {
        return do_transition(target);
    }

    /* Check transition table */
    if (allowed[current_mode][target] == 0U) {
        return MODE_ERR_ILLEGAL;
    }

    return do_transition(target);
}

int32_t mode_request_safe(void)
{
    if (current_mode == MODE_SAFE) {
        return MODE_OK;
    }
    return do_transition(MODE_SAFE);
}

obc_mode_t mode_get_current(void)
{
    return current_mode;
}

obc_mode_t mode_get_previous(void)
{
    return previous_mode;
}

uint32_t mode_get_transition_count(void)
{
    return transition_count;
}
