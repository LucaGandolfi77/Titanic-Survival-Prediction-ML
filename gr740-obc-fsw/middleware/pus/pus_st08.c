/**
 * @file pus_st08.c
 * @brief PUS Service 8 — Function Management implementation.
 *
 * Dispatch table for on-board function execution via TC(8,1).
 * Registered functions are looked up by ID and invoked with
 * provided arguments. No dynamic allocation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st08.h"

/* ── Function registry ─────────────────────────────────────────────────── */
typedef struct {
    uint16_t            func_id;
    pus_func_handler_t  handler;
    uint8_t             valid;
} func_entry_t;

static func_entry_t func_table[PUS_ST08_MAX_FUNCS];
static uint8_t st08_init_done = 0U;

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st08_init(void)
{
    uint32_t i;

    for (i = 0U; i < PUS_ST08_MAX_FUNCS; i++) {
        func_table[i].func_id = 0U;
        func_table[i].handler = (pus_func_handler_t)0;
        func_table[i].valid = 0U;
    }

    st08_init_done = 1U;
    return PUS_ST08_OK;
}

int32_t pus_st08_register(uint16_t func_id, pus_func_handler_t handler)
{
    uint32_t i;

    if (st08_init_done == 0U) {
        return PUS_ST08_ERR_PARAM;
    }
    if (handler == (pus_func_handler_t)0) {
        return PUS_ST08_ERR_PARAM;
    }

    /* Check if already registered */
    for (i = 0U; i < PUS_ST08_MAX_FUNCS; i++) {
        if ((func_table[i].valid != 0U) && (func_table[i].func_id == func_id)) {
            func_table[i].handler = handler;
            return PUS_ST08_OK;
        }
    }

    /* Find free slot */
    for (i = 0U; i < PUS_ST08_MAX_FUNCS; i++) {
        if (func_table[i].valid == 0U) {
            func_table[i].func_id = func_id;
            func_table[i].handler = handler;
            func_table[i].valid = 1U;
            return PUS_ST08_OK;
        }
    }

    return PUS_ST08_ERR_FULL;
}

int32_t pus_st08_execute(uint16_t func_id, const uint8_t *args, uint32_t arg_len)
{
    uint32_t i;

    if (st08_init_done == 0U) {
        return PUS_ST08_ERR_PARAM;
    }

    for (i = 0U; i < PUS_ST08_MAX_FUNCS; i++) {
        if ((func_table[i].valid != 0U) && (func_table[i].func_id == func_id)) {
            if (func_table[i].handler != (pus_func_handler_t)0) {
                return func_table[i].handler(args, arg_len);
            }
            return PUS_ST08_ERR_EXEC;
        }
    }

    return PUS_ST08_ERR_NOT_FOUND;
}

int32_t pus_st08_process(const uint8_t *data, uint32_t len)
{
    uint16_t func_id;

    if (data == (const uint8_t *)0) {
        return PUS_ST08_ERR_PARAM;
    }
    if (len < 2U) {
        return PUS_ST08_ERR_PARAM;
    }

    /* Extract function ID (big-endian) */
    func_id = (uint16_t)(((uint16_t)data[0] << 8U) | (uint16_t)data[1]);

    /* Execute with remaining bytes as arguments */
    return pus_st08_execute(func_id, &data[2], len - 2U);
}
