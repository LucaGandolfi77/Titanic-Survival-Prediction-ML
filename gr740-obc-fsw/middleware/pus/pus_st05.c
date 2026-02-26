/**
 * @file pus_st05.c
 * @brief PUS Service 5 — Event Reporting implementation.
 *
 * Generates event TM(5,x) reports per ECSS-E-ST-70-41C.
 * Events are rate-limited (min 100ms between same event) and
 * can be individually enabled/disabled. Four severity levels
 * map to subtypes 1-4 respectively.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st05.h"
#include "../ccsds/space_packet.h"

extern int32_t router_send_tm(const ccsds_packet_t *pkt);
extern uint32_t bsp_get_uptime_ms(void);

/* ── Module state ──────────────────────────────────────────────────────── */
static uint16_t st05_apid      = 0U;
static uint8_t  st05_init_done = 0U;

/** Per-event state */
typedef struct {
    uint8_t  enabled;       /**< Reporting enabled         */
    uint32_t last_time_ms;  /**< Last report time          */
    uint32_t count;         /**< Total raise count         */
} evt_state_t;

static evt_state_t evt_states[PUS_ST05_MAX_EVENTS];

/** Severity counters */
static uint32_t sev_counters[4];

/** Minimum interval between same event reports (ms) */
#define EVT_MIN_INTERVAL_MS  100U

#define PUS_VERSION_C   0x20U
#define PUS_SVC_TYPE_05 5U

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st05_init(uint16_t apid)
{
    uint32_t i;

    if (apid > 0x7FFU) {
        return PUS_ST05_ERR_PARAM;
    }

    st05_apid = apid;

    for (i = 0U; i < PUS_ST05_MAX_EVENTS; i++) {
        evt_states[i].enabled = 1U; /* All enabled by default */
        evt_states[i].last_time_ms = 0U;
        evt_states[i].count = 0U;
    }

    for (i = 0U; i < 4U; i++) {
        sev_counters[i] = 0U;
    }

    st05_init_done = 1U;
    return PUS_ST05_OK;
}

int32_t pus_st05_raise(uint16_t event_id, uint8_t severity,
                         const uint8_t *aux_data, uint8_t aux_len)
{
    ccsds_packet_t tm_pkt;
    uint8_t data[48]; /* PUS sec hdr + event ID + aux */
    uint32_t pos;
    uint16_t seq;
    uint32_t time_ms;
    uint32_t time_s;
    uint8_t  subtype;
    int32_t  ret;
    uint32_t i;

    if (st05_init_done == 0U) {
        return PUS_ST05_ERR_PARAM;
    }
    if (event_id >= PUS_ST05_MAX_EVENTS) {
        return PUS_ST05_ERR_PARAM;
    }
    if (severity > EVT_SEV_HIGH) {
        return PUS_ST05_ERR_PARAM;
    }
    if (aux_len > PUS_ST05_MAX_AUX) {
        return PUS_ST05_ERR_PARAM;
    }

    /* Check if event is enabled */
    if (evt_states[event_id].enabled == 0U) {
        return PUS_ST05_ERR_DISABLED;
    }

    /* Rate limiting */
    time_ms = bsp_get_uptime_ms();
    if ((time_ms - evt_states[event_id].last_time_ms) < EVT_MIN_INTERVAL_MS) {
        return PUS_ST05_ERR_RATE;
    }

    evt_states[event_id].last_time_ms = time_ms;
    evt_states[event_id].count++;
    sev_counters[severity]++;

    /* Map severity to subtype */
    subtype = (uint8_t)(severity + 1U); /* INFO=1, LOW=2, MED=3, HIGH=4 */

    pos = 0U;

    /* PUS-C secondary header */
    data[pos] = PUS_VERSION_C;    pos++;
    data[pos] = PUS_SVC_TYPE_05;  pos++;
    data[pos] = subtype;          pos++;
    data[pos] = 0x00U;            pos++; /* Source ID high */
    data[pos] = 0x01U;            pos++; /* Source ID low  */
    /* Timestamp CUC 4 bytes */
    time_s = time_ms / 1000U;
    data[pos] = (uint8_t)((time_s >> 24U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((time_s >> 16U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((time_s >> 8U)  & 0xFFU); pos++;
    data[pos] = (uint8_t)(time_s & 0xFFU);          pos++;

    /* Event ID (2 bytes) */
    data[pos] = (uint8_t)((event_id >> 8U) & 0xFFU); pos++;
    data[pos] = (uint8_t)(event_id & 0xFFU);         pos++;

    /* Auxiliary data */
    if ((aux_data != (const uint8_t *)0) && (aux_len > 0U)) {
        for (i = 0U; i < (uint32_t)aux_len; i++) {
            data[pos] = aux_data[i];
            pos++;
        }
    }

    /* Build CCSDS packet */
    ret = ccsds_init_packet(&tm_pkt, CCSDS_TYPE_TM, st05_apid,
                             CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    if (ret != CCSDS_OK) {
        return PUS_ST05_ERR_PARAM;
    }

    seq = ccsds_next_seq_count(st05_apid);
    tm_pkt.header.pkt_seq_ctrl = (uint16_t)(
        (tm_pkt.header.pkt_seq_ctrl & 0xC000U) | (seq & 0x3FFFU)
    );

    ret = ccsds_set_data(&tm_pkt, data, pos);
    if (ret != CCSDS_OK) {
        return PUS_ST05_ERR_PARAM;
    }

    ret = router_send_tm(&tm_pkt);
    (void)ret;

    return PUS_ST05_OK;
}

int32_t pus_st05_enable(uint16_t event_id)
{
    if (event_id >= PUS_ST05_MAX_EVENTS) {
        return PUS_ST05_ERR_PARAM;
    }
    evt_states[event_id].enabled = 1U;
    return PUS_ST05_OK;
}

int32_t pus_st05_disable(uint16_t event_id)
{
    if (event_id >= PUS_ST05_MAX_EVENTS) {
        return PUS_ST05_ERR_PARAM;
    }
    evt_states[event_id].enabled = 0U;
    return PUS_ST05_OK;
}

void pus_st05_get_counters(uint32_t *info_cnt, uint32_t *low_cnt,
                            uint32_t *med_cnt, uint32_t *high_cnt)
{
    if (info_cnt != (uint32_t *)0) { *info_cnt = sev_counters[0]; }
    if (low_cnt  != (uint32_t *)0) { *low_cnt  = sev_counters[1]; }
    if (med_cnt  != (uint32_t *)0) { *med_cnt  = sev_counters[2]; }
    if (high_cnt != (uint32_t *)0) { *high_cnt = sev_counters[3]; }
}
