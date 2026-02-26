/**
 * @file parameter_table.c
 * @brief Parameter Table implementation.
 *
 * In-RAM table backed by NVM via nvm_manager.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "parameter_table.h"
#include "nvm_manager.h"

/* ── NVM region for parameter table ────────────────────────────────────── */
#define PARAM_NVM_REGION_ID   0U
#define PARAM_NVM_OFFSET      0x00000000U
#define PARAM_NVM_SIZE        (PARAM_MAX_ENTRIES * (PARAM_MAX_VALUE_SIZE + 8U))

/* ── Entry descriptor ──────────────────────────────────────────────────── */
typedef struct {
    uint16_t    param_id;
    param_type_t type;
    uint8_t     value[PARAM_MAX_VALUE_SIZE];
    uint8_t     default_val[PARAM_MAX_VALUE_SIZE];
    uint8_t     size;
    uint8_t     defined;
} param_entry_t;

/* ── Module state ──────────────────────────────────────────────────────── */
static param_entry_t entries[PARAM_MAX_ENTRIES];
static uint32_t      entry_count = 0U;
static uint8_t       pt_init_done = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

static param_entry_t *find_entry(uint16_t param_id)
{
    uint32_t i;
    for (i = 0U; i < entry_count; i++) {
        if ((entries[i].defined != 0U) &&
            (entries[i].param_id == param_id)) {
            return &entries[i];
        }
    }
    return (param_entry_t *)0;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t param_table_init(void)
{
    uint32_t i;

    for (i = 0U; i < PARAM_MAX_ENTRIES; i++) {
        entries[i].defined  = 0U;
        entries[i].param_id = 0U;
        entries[i].size     = 0U;
    }
    entry_count = 0U;

    /* Register NVM region for persistence */
    (void)nvm_register_region(PARAM_NVM_REGION_ID, NVM_STORE_MRAM,
                               PARAM_NVM_OFFSET, PARAM_NVM_SIZE);

    pt_init_done = 1U;
    return PARAM_OK;
}

int32_t param_define(uint16_t param_id, param_type_t type,
                      const uint8_t *default_val, uint8_t size)
{
    param_entry_t *e;
    uint8_t i;

    if (size > PARAM_MAX_VALUE_SIZE) {
        return PARAM_ERR_PARAM;
    }
    if (default_val == (const uint8_t *)0) {
        return PARAM_ERR_PARAM;
    }

    /* Check for existing entry */
    e = find_entry(param_id);
    if (e != (param_entry_t *)0) {
        /* Update existing */
        e->type = type;
        e->size = size;
        for (i = 0U; i < size; i++) {
            e->default_val[i] = default_val[i];
            e->value[i]       = default_val[i];
        }
        return PARAM_OK;
    }

    if (entry_count >= PARAM_MAX_ENTRIES) {
        return PARAM_ERR_FULL;
    }

    e = &entries[entry_count];
    e->param_id = param_id;
    e->type     = type;
    e->size     = size;
    e->defined  = 1U;

    for (i = 0U; i < size; i++) {
        e->default_val[i] = default_val[i];
        e->value[i]       = default_val[i];
    }

    entry_count++;
    return PARAM_OK;
}

int32_t param_set(uint16_t param_id, const uint8_t *value, uint8_t size)
{
    param_entry_t *e;
    uint8_t i;

    e = find_entry(param_id);
    if (e == (param_entry_t *)0) {
        return PARAM_ERR_NOT_FOUND;
    }
    if (size != e->size) {
        return PARAM_ERR_PARAM;
    }
    if (value == (const uint8_t *)0) {
        return PARAM_ERR_PARAM;
    }

    for (i = 0U; i < size; i++) {
        e->value[i] = value[i];
    }

    return PARAM_OK;
}

int32_t param_get(uint16_t param_id, uint8_t *value, uint8_t *size)
{
    param_entry_t *e;
    uint8_t i;

    if ((value == (uint8_t *)0) || (size == (uint8_t *)0)) {
        return PARAM_ERR_PARAM;
    }

    e = find_entry(param_id);
    if (e == (param_entry_t *)0) {
        return PARAM_ERR_NOT_FOUND;
    }

    *size = e->size;
    for (i = 0U; i < e->size; i++) {
        value[i] = e->value[i];
    }

    return PARAM_OK;
}

int32_t param_save_to_nvm(void)
{
    /* Serialise all entries into a contiguous buffer and write to NVM.
     * Format per entry: [2] param_id + [1] size + [N] value            */
    uint8_t  buf[PARAM_NVM_SIZE];
    uint32_t pos = 0U;
    uint32_t i;
    uint8_t  j;

    for (i = 0U; i < entry_count; i++) {
        if (entries[i].defined == 0U) {
            continue;
        }
        if ((pos + 3U + (uint32_t)entries[i].size) > PARAM_NVM_SIZE) {
            break;
        }
        buf[pos] = (uint8_t)((entries[i].param_id >> 8U) & 0xFFU);
        pos++;
        buf[pos] = (uint8_t)(entries[i].param_id & 0xFFU);
        pos++;
        buf[pos] = entries[i].size;
        pos++;
        for (j = 0U; j < entries[i].size; j++) {
            buf[pos] = entries[i].value[j];
            pos++;
        }
    }

    return nvm_write(PARAM_NVM_REGION_ID, buf, pos);
}

int32_t param_load_from_nvm(void)
{
    uint8_t  buf[PARAM_NVM_SIZE];
    uint32_t actual_len = 0U;
    uint32_t pos = 0U;
    int32_t  ret;

    ret = nvm_read(PARAM_NVM_REGION_ID, buf, PARAM_NVM_SIZE, &actual_len);
    if (ret != NVM_OK) {
        return PARAM_ERR_NVM;
    }

    while ((pos + 3U) <= actual_len) {
        uint16_t pid;
        uint8_t  sz;
        param_entry_t *e;

        pid = (uint16_t)(((uint16_t)buf[pos] << 8U) | (uint16_t)buf[pos + 1U]);
        pos += 2U;
        sz = buf[pos];
        pos++;

        if ((pos + (uint32_t)sz) > actual_len) {
            break;
        }

        e = find_entry(pid);
        if ((e != (param_entry_t *)0) && (e->size == sz)) {
            uint8_t j;
            for (j = 0U; j < sz; j++) {
                e->value[j] = buf[pos + (uint32_t)j];
            }
        }
        pos += (uint32_t)sz;
    }

    return PARAM_OK;
}

int32_t param_reset_defaults(void)
{
    uint32_t i;
    uint8_t  j;

    for (i = 0U; i < entry_count; i++) {
        if (entries[i].defined != 0U) {
            for (j = 0U; j < entries[i].size; j++) {
                entries[i].value[j] = entries[i].default_val[j];
            }
        }
    }
    return PARAM_OK;
}

uint32_t param_count(void)
{
    return entry_count;
}
