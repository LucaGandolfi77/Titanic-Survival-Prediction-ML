/**
 * @file pus_st03.c
 * @brief PUS Service 3 — Housekeeping implementation.
 *
 * Manages HK report structure definitions, periodic generation,
 * and on-demand reports. Collects parameter values via registered
 * reader callback and packages into TM(3,25) CCSDS packets.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st03.h"
#include "../ccsds/space_packet.h"

extern int32_t router_send_tm(const ccsds_packet_t *pkt);
extern uint32_t bsp_get_uptime_ms(void);

/* ── Module state ──────────────────────────────────────────────────────── */
static uint16_t         st03_apid      = 0U;
static hk_param_reader_t st03_reader   = (hk_param_reader_t)0;
static hk_definition_t  hk_defs[PUS_ST03_MAX_SID];
static uint8_t          st03_init_done = 0U;

#define PUS_VERSION_C   0x20U
#define PUS_SVC_TYPE_03 3U

/* ── Find definition by SID ────────────────────────────────────────────── */
static hk_definition_t *find_def(uint16_t sid)
{
    uint32_t i;
    for (i = 0U; i < PUS_ST03_MAX_SID; i++) {
        if ((hk_defs[i].valid != 0U) && (hk_defs[i].sid == sid)) {
            return &hk_defs[i];
        }
    }
    return (hk_definition_t *)0;
}

/* ── Find free slot ────────────────────────────────────────────────────── */
static hk_definition_t *find_free(void)
{
    uint32_t i;
    for (i = 0U; i < PUS_ST03_MAX_SID; i++) {
        if (hk_defs[i].valid == 0U) {
            return &hk_defs[i];
        }
    }
    return (hk_definition_t *)0;
}

/* ── Generate a HK report for a given definition ──────────────────────── */
static int32_t generate_report(const hk_definition_t *def)
{
    ccsds_packet_t tm_pkt;
    uint8_t data[512]; /* PUS sec hdr + SID + param data */
    uint32_t pos;
    uint32_t param_len;
    uint16_t seq;
    uint32_t time_s;
    uint32_t i;
    int32_t  ret;

    if ((def == (const hk_definition_t *)0) ||
        (st03_reader == (hk_param_reader_t)0)) {
        return PUS_ST03_ERR_PARAM;
    }

    pos = 0U;

    /* PUS-C secondary header */
    data[pos] = PUS_VERSION_C;
    pos++;
    data[pos] = PUS_SVC_TYPE_03;
    pos++;
    data[pos] = PUS_ST03_HK_REPORT; /* Subtype = 25 */
    pos++;
    /* Source ID */
    data[pos] = 0x00U;
    pos++;
    data[pos] = 0x01U;
    pos++;
    /* Timestamp CUC 4 bytes */
    time_s = bsp_get_uptime_ms() / 1000U;
    data[pos] = (uint8_t)((time_s >> 24U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((time_s >> 16U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((time_s >> 8U)  & 0xFFU); pos++;
    data[pos] = (uint8_t)(time_s & 0xFFU);          pos++;

    /* Structure ID (2 bytes) */
    data[pos] = (uint8_t)((def->sid >> 8U) & 0xFFU); pos++;
    data[pos] = (uint8_t)(def->sid & 0xFFU);         pos++;

    /* Collect parameter values */
    for (i = 0U; i < (uint32_t)def->num_params; i++) {
        param_len = 0U;
        ret = st03_reader(def->param_ids[i], &data[pos], &param_len);
        if (ret != 0) {
            /* On read error, fill with 0xFF */
            param_len = (uint32_t)def->param_sizes[i];
            if ((pos + param_len) > sizeof(data)) {
                break;
            }
            {
                uint32_t j;
                for (j = 0U; j < param_len; j++) {
                    data[pos + j] = 0xFFU;
                }
            }
        }
        if (param_len == 0U) {
            param_len = (uint32_t)def->param_sizes[i];
        }
        if ((pos + param_len) > sizeof(data)) {
            break;
        }
        pos += param_len;
    }

    /* Build CCSDS TM packet */
    ret = ccsds_init_packet(&tm_pkt, CCSDS_TYPE_TM, st03_apid,
                             CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    if (ret != CCSDS_OK) {
        return PUS_ST03_ERR_PARAM;
    }

    seq = ccsds_next_seq_count(st03_apid);
    tm_pkt.header.pkt_seq_ctrl = (uint16_t)(
        (tm_pkt.header.pkt_seq_ctrl & 0xC000U) | (seq & 0x3FFFU)
    );

    ret = ccsds_set_data(&tm_pkt, data, pos);
    if (ret != CCSDS_OK) {
        return PUS_ST03_ERR_PARAM;
    }

    ret = router_send_tm(&tm_pkt);
    (void)ret;

    return PUS_ST03_OK;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st03_init(uint16_t apid, hk_param_reader_t reader)
{
    uint32_t i;

    if (apid > 0x7FFU) {
        return PUS_ST03_ERR_PARAM;
    }
    if (reader == (hk_param_reader_t)0) {
        return PUS_ST03_ERR_PARAM;
    }

    st03_apid = apid;
    st03_reader = reader;

    for (i = 0U; i < PUS_ST03_MAX_SID; i++) {
        hk_defs[i].valid = 0U;
    }

    st03_init_done = 1U;
    return PUS_ST03_OK;
}

int32_t pus_st03_define(uint16_t sid, const uint16_t *param_ids,
                         const uint8_t *param_sizes, uint8_t num_params,
                         uint16_t period_ms)
{
    hk_definition_t *def;
    uint32_t i;

    if (st03_init_done == 0U) {
        return PUS_ST03_ERR_PARAM;
    }
    if ((param_ids == (const uint16_t *)0) ||
        (param_sizes == (const uint8_t *)0)) {
        return PUS_ST03_ERR_PARAM;
    }
    if (num_params > PUS_ST03_MAX_PARAMS) {
        return PUS_ST03_ERR_PARAM;
    }

    /* Check if SID already exists */
    def = find_def(sid);
    if (def == (hk_definition_t *)0) {
        def = find_free();
        if (def == (hk_definition_t *)0) {
            return PUS_ST03_ERR_FULL;
        }
    }

    def->sid = sid;
    def->num_params = num_params;
    def->period_ms = period_ms;
    def->enabled = 0U;
    def->last_time_ms = 0U;

    for (i = 0U; i < (uint32_t)num_params; i++) {
        def->param_ids[i]   = param_ids[i];
        def->param_sizes[i] = param_sizes[i];
    }

    def->valid = 1U;
    return PUS_ST03_OK;
}

int32_t pus_st03_delete(uint16_t sid)
{
    hk_definition_t *def = find_def(sid);
    if (def == (hk_definition_t *)0) {
        return PUS_ST03_ERR_NOT_FOUND;
    }
    def->valid = 0U;
    return PUS_ST03_OK;
}

int32_t pus_st03_enable(uint16_t sid)
{
    hk_definition_t *def = find_def(sid);
    if (def == (hk_definition_t *)0) {
        return PUS_ST03_ERR_NOT_FOUND;
    }
    def->enabled = 1U;
    return PUS_ST03_OK;
}

int32_t pus_st03_disable(uint16_t sid)
{
    hk_definition_t *def = find_def(sid);
    if (def == (hk_definition_t *)0) {
        return PUS_ST03_ERR_NOT_FOUND;
    }
    def->enabled = 0U;
    return PUS_ST03_OK;
}

int32_t pus_st03_report(uint16_t sid)
{
    hk_definition_t *def = find_def(sid);
    if (def == (hk_definition_t *)0) {
        return PUS_ST03_ERR_NOT_FOUND;
    }
    return generate_report(def);
}

void pus_st03_tick(uint32_t current_time_ms)
{
    uint32_t i;
    uint32_t elapsed;

    if (st03_init_done == 0U) {
        return;
    }

    for (i = 0U; i < PUS_ST03_MAX_SID; i++) {
        if ((hk_defs[i].valid != 0U) && (hk_defs[i].enabled != 0U) &&
            (hk_defs[i].period_ms > 0U)) {
            elapsed = current_time_ms - hk_defs[i].last_time_ms;
            if (elapsed >= (uint32_t)hk_defs[i].period_ms) {
                (void)generate_report(&hk_defs[i]);
                hk_defs[i].last_time_ms = current_time_ms;
            }
        }
    }
}
