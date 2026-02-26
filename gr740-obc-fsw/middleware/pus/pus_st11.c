/**
 * @file pus_st11.c
 * @brief PUS Service 11 — Time-Based Scheduling implementation.
 *
 * Manages a table of time-tagged TC packets. On each tick, due
 * activities are dispatched in time order. Individual or bulk
 * insert/delete/enable/disable supported.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st11.h"

/* ── Module state ──────────────────────────────────────────────────────── */
static sched_entry_t sched_table[PUS_ST11_MAX_SCHED];
static uint8_t       sched_enabled = 0U;
static uint16_t      next_request_id = 1U;
static uint8_t       st11_init_done = 0U;

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st11_init(void)
{
    uint32_t i;

    for (i = 0U; i < PUS_ST11_MAX_SCHED; i++) {
        sched_table[i].valid = 0U;
    }

    sched_enabled = 0U;
    next_request_id = 1U;
    st11_init_done = 1U;

    return PUS_ST11_OK;
}

int32_t pus_st11_insert(uint32_t exec_time, const ccsds_packet_t *tc_pkt)
{
    uint32_t i;
    uint32_t j;

    if (st11_init_done == 0U) {
        return PUS_ST11_ERR_PARAM;
    }
    if (tc_pkt == (const ccsds_packet_t *)0) {
        return PUS_ST11_ERR_PARAM;
    }

    /* Find free slot */
    for (i = 0U; i < PUS_ST11_MAX_SCHED; i++) {
        if (sched_table[i].valid == 0U) {
            sched_table[i].exec_time = exec_time;
            sched_table[i].request_id = next_request_id;
            next_request_id++;
            if (next_request_id == 0U) {
                next_request_id = 1U; /* Skip 0 */
            }

            /* Copy TC packet */
            sched_table[i].tc_packet.header = tc_pkt->header;
            sched_table[i].tc_packet.data_len = tc_pkt->data_len;
            sched_table[i].tc_packet.crc = tc_pkt->crc;
            for (j = 0U; j < tc_pkt->data_len; j++) {
                sched_table[i].tc_packet.data[j] = tc_pkt->data[j];
            }

            sched_table[i].valid = 1U;
            return PUS_ST11_OK;
        }
    }

    return PUS_ST11_ERR_FULL;
}

int32_t pus_st11_delete(uint16_t request_id)
{
    uint32_t i;

    if (st11_init_done == 0U) {
        return PUS_ST11_ERR_PARAM;
    }

    for (i = 0U; i < PUS_ST11_MAX_SCHED; i++) {
        if ((sched_table[i].valid != 0U) &&
            (sched_table[i].request_id == request_id)) {
            sched_table[i].valid = 0U;
            return PUS_ST11_OK;
        }
    }

    return PUS_ST11_ERR_NOT_FOUND;
}

void pus_st11_enable(void)
{
    sched_enabled = 1U;
}

void pus_st11_disable(void)
{
    sched_enabled = 0U;
}

void pus_st11_reset(void)
{
    uint32_t i;

    for (i = 0U; i < PUS_ST11_MAX_SCHED; i++) {
        sched_table[i].valid = 0U;
    }
}

void pus_st11_tick(uint32_t current_time,
                    int32_t (*handler)(const ccsds_packet_t *))
{
    uint32_t i;
    uint32_t earliest_idx;
    uint32_t earliest_time;
    uint8_t  found;

    if ((st11_init_done == 0U) || (sched_enabled == 0U)) {
        return;
    }
    if (handler == (int32_t (*)(const ccsds_packet_t *))0) {
        return;
    }

    /* Process all due activities in time order */
    for (;;) {
        /* Find earliest due activity */
        found = 0U;
        earliest_time = 0xFFFFFFFFU;
        earliest_idx = 0U;

        for (i = 0U; i < PUS_ST11_MAX_SCHED; i++) {
            if ((sched_table[i].valid != 0U) &&
                (sched_table[i].exec_time <= current_time)) {
                if (sched_table[i].exec_time < earliest_time) {
                    earliest_time = sched_table[i].exec_time;
                    earliest_idx = i;
                    found = 1U;
                }
            }
        }

        if (found == 0U) {
            break; /* No more due activities */
        }

        /* Dispatch */
        (void)handler(&sched_table[earliest_idx].tc_packet);

        /* Mark as executed */
        sched_table[earliest_idx].valid = 0U;
    }
}

uint32_t pus_st11_count(void)
{
    uint32_t i;
    uint32_t count = 0U;

    for (i = 0U; i < PUS_ST11_MAX_SCHED; i++) {
        if (sched_table[i].valid != 0U) {
            count++;
        }
    }

    return count;
}
