/**
 * @file pus_st17.c
 * @brief PUS Service 17 — Test Connection implementation.
 *
 * Responds to TC(17,1) "Are you alive?" with TM(17,2) "I am alive".
 * Report carries PUS-C secondary header with current time.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st17.h"
#include "../ccsds/space_packet.h"

extern int32_t  router_send_tm(const ccsds_packet_t *pkt);
extern uint32_t bsp_get_uptime_ms(void);

/* ── Module state ──────────────────────────────────────────────────────── */
static uint16_t st17_apid      = 0U;
static uint8_t  st17_init_done = 0U;

#define PUS_VERSION_C    0x20U
#define PUS_SVC_TYPE_17  17U

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st17_init(uint16_t apid)
{
    if (apid > 0x7FFU) {
        return PUS_ST17_ERR_PARAM;
    }

    st17_apid = apid;
    st17_init_done = 1U;
    return PUS_ST17_OK;
}

int32_t pus_st17_handle(void)
{
    ccsds_packet_t tm_pkt;
    uint8_t data[16];
    uint32_t pos;
    uint16_t seq;
    uint32_t time_s;
    int32_t  ret;

    if (st17_init_done == 0U) {
        return PUS_ST17_ERR_PARAM;
    }

    pos = 0U;

    /* PUS-C secondary header */
    data[pos] = PUS_VERSION_C;                  pos++;
    data[pos] = PUS_SVC_TYPE_17;                pos++;
    data[pos] = PUS_ST17_I_AM_ALIVE;            pos++;
    data[pos] = 0x00U;                          pos++; /* Source ID high */
    data[pos] = 0x01U;                          pos++; /* Source ID low  */

    /* Timestamp CUC coarse 4 bytes */
    time_s = bsp_get_uptime_ms() / 1000U;
    data[pos] = (uint8_t)((time_s >> 24U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((time_s >> 16U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((time_s >> 8U)  & 0xFFU); pos++;
    data[pos] = (uint8_t)(time_s & 0xFFU);          pos++;

    /* No additional data for are-you-alive response */

    /* Build CCSDS TM packet */
    ret = ccsds_init_packet(&tm_pkt, CCSDS_TYPE_TM, st17_apid,
                             CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    if (ret != CCSDS_OK) {
        return PUS_ST17_ERR_PARAM;
    }

    seq = ccsds_next_seq_count(st17_apid);
    tm_pkt.header.pkt_seq_ctrl = (uint16_t)(
        (tm_pkt.header.pkt_seq_ctrl & 0xC000U) | (seq & 0x3FFFU)
    );

    ret = ccsds_set_data(&tm_pkt, data, pos);
    if (ret != CCSDS_OK) {
        return PUS_ST17_ERR_PARAM;
    }

    ret = router_send_tm(&tm_pkt);
    (void)ret;

    return PUS_ST17_OK;
}
