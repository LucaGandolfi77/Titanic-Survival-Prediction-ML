/**
 * @file mode_manager.h
 * @brief Mode Manager — operational mode FSM API.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef MODE_MANAGER_H
#define MODE_MANAGER_H

#include <stdint.h>
#include "modes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define MODE_OK              0
#define MODE_ERR_PARAM      -1
#define MODE_ERR_ILLEGAL    -2   /**< Transition not allowed          */

/* ── Mode transition callback ──────────────────────────────────────────── */
/**
 * @brief Called on entry to a new mode.
 * @param[in] old_mode  Previous mode.
 * @param[in] new_mode  New mode being entered.
 * @return 0 on success, negative on error (transition aborted).
 */
typedef int32_t (*mode_entry_fn_t)(obc_mode_t old_mode, obc_mode_t new_mode);

/**
 * @brief Called on exit from current mode.
 */
typedef void (*mode_exit_fn_t)(obc_mode_t current, obc_mode_t next);

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the mode manager.
 *
 * Starts in MODE_BOOT.
 *
 * @return MODE_OK on success.
 */
int32_t mode_manager_init(void);

/**
 * @brief Register entry/exit callbacks for a mode.
 *
 * @param[in] mode      Target mode.
 * @param[in] on_entry  Entry callback (may be NULL).
 * @param[in] on_exit   Exit callback (may be NULL).
 * @return MODE_OK on success.
 */
int32_t mode_register_callbacks(obc_mode_t mode,
                                 mode_entry_fn_t on_entry,
                                 mode_exit_fn_t on_exit);

/**
 * @brief Request a mode transition.
 *
 * Validates the transition against the allowed FSM:
 *   BOOT   → SAFE
 *   SAFE   → NOMINAL, DETUMBLING
 *   NOMINAL→ SCIENCE, ECLIPSE, SAFE
 *   SCIENCE→ NOMINAL, SAFE
 *   ECLIPSE→ NOMINAL, SAFE
 *   DETUMBLING → SAFE
 *   Any    → SAFE  (always allowed)
 *
 * @param[in] target  Requested mode.
 * @return MODE_OK on success, MODE_ERR_ILLEGAL if not allowed.
 */
int32_t mode_request_transition(obc_mode_t target);

/**
 * @brief Force transition to SAFE mode (FDIR escalation).
 *
 * Always succeeds regardless of current mode.
 *
 * @return MODE_OK.
 */
int32_t mode_request_safe(void);

/**
 * @brief Get the current operating mode.
 * @return Current mode.
 */
obc_mode_t mode_get_current(void);

/**
 * @brief Get the previous mode (before last transition).
 * @return Previous mode.
 */
obc_mode_t mode_get_previous(void);

/**
 * @brief Get total number of mode transitions since boot.
 * @return Transition counter.
 */
uint32_t mode_get_transition_count(void);

#ifdef __cplusplus
}
#endif

#endif /* MODE_MANAGER_H */
