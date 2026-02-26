/**
 * @file pus_st01.c
 * @brief PUS Service 1 — Request Verification implementation.
 *
 * Generates TM verification reports per ECSS-E-ST-70-41C.
 * Each report echoes the TC packet identification (APID + sequence count)
 * and optionally includes failure codes.
 *
 * PUS-C secondary header format (data field):
 *   Byte 0:    PUS Version (0x2x = PUS-C, upper nibble=version)
 *   Byte 1:    Service Type
 *   Byte 2:    Service Subtype
 *   Byte 3-4:  Destination ID (for TC) / Source ID (for TM)
 *   Byte 5-8:  Time stamp (CUC 4-byte coarse)
 *
 * Verification report data:
 *   Byte 0-1:  TC Packet ID
 *   Byte 2-3:  TC Packet Sequence Control
 *   (For failure reports):
 *   Byte 4-5:  Error Code
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st01.h"

/* Forward declaration — implemented by packet router */
extern int32_t router_send_tm(const ccsds_packet_t *pkt);
extern uint32_t bsp_get_uptime_ms(void);

/* ── Module state ──────────────────────────────────────────────────────── */
static uint16_t st01_apid     = 0U;
static uint8_t  st01_init_done = 0U;

/** PUS-C version number (upper nibble) */
#define PUS_VERSION_C   0x20U

/** PUS Service 1 type */
#define PUS_SVC_TYPE_01 1U

/* ── PUS-C TM secondary header size ────────────────────────────────────── */
#define PUS_TM_SEC_HDR_SIZE  9U  /* version(1)+type(1)+subtype(1)+destid(2)+time(4) */

/* ── Build verification TM report ──────────────────────────────────────── */
static int32_t build_verif_report(uint8_t subtype,
                                   const ccsds_packet_t *tc_pkt,
                                   uint16_t err_code,
                                   uint8_t  has_error)
{
    ccsds_packet_t tm_pkt;
    uint8_t data[PUS_TM_SEC_HDR_SIZE + 6]; /* sec hdr + TC ID + seq + err */
    uint32_t pos;
    uint16_t seq;
    uint32_t time_s;
    int32_t  ret;

    if (st01_init_done == 0U) {
        return PUS_ST01_ERR_PARAM;
    }
    if (tc_pkt == (const ccsds_packet_t *)0) {
        return PUS_ST01_ERR_PARAM;
    }

    pos = 0U;

    /* PUS-C secondary header */
    data[pos] = PUS_VERSION_C;    /* Version */
    pos++;
    data[pos] = PUS_SVC_TYPE_01;  /* Service Type = 1 */
    pos++;
    data[pos] = subtype;          /* Service Subtype */
    pos++;
    /* Source ID (2 bytes) — this OBC */
    data[pos] = 0x00U;
    pos++;
    data[pos] = 0x01U;
    pos++;
    /* Timestamp: CUC coarse 4 bytes (seconds since epoch) */
    time_s = bsp_get_uptime_ms() / 1000U;
    data[pos] = (uint8_t)((time_s >> 24U) & 0xFFU);
    pos++;
    data[pos] = (uint8_t)((time_s >> 16U) & 0xFFU);
    pos++;
    data[pos] = (uint8_t)((time_s >> 8U) & 0xFFU);
    pos++;
    data[pos] = (uint8_t)(time_s & 0xFFU);
    pos++;

    /* TC Packet Identification (2 bytes) */
    data[pos] = (uint8_t)((tc_pkt->header.pkt_id >> 8U) & 0xFFU);
    pos++;
    data[pos] = (uint8_t)(tc_pkt->header.pkt_id & 0xFFU);
    pos++;

    /* TC Packet Sequence Control (2 bytes) */
    data[pos] = (uint8_t)((tc_pkt->header.pkt_seq_ctrl >> 8U) & 0xFFU);
    pos++;
    data[pos] = (uint8_t)(tc_pkt->header.pkt_seq_ctrl & 0xFFU);
    pos++;

    /* Error code (for failure reports only) */
    if (has_error != 0U) {
        data[pos] = (uint8_t)((err_code >> 8U) & 0xFFU);
        pos++;
        data[pos] = (uint8_t)(err_code & 0xFFU);
        pos++;
    }

    /* Build CCSDS packet */
    ret = ccsds_init_packet(&tm_pkt, CCSDS_TYPE_TM, st01_apid,
                             CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    if (ret != CCSDS_OK) {
        return PUS_ST01_ERR_PARAM;
    }

    /* Set sequence count */
    seq = ccsds_next_seq_count(st01_apid);
    tm_pkt.header.pkt_seq_ctrl = (uint16_t)(
        (tm_pkt.header.pkt_seq_ctrl & 0xC000U) | (seq & 0x3FFFU)
    );

    ret = ccsds_set_data(&tm_pkt, data, pos);
    if (ret != CCSDS_OK) {
        return PUS_ST01_ERR_PARAM;
    }

    /* Send via router */
    ret = router_send_tm(&tm_pkt);
    (void)ret; /* Best-effort delivery for verification reports */

    return PUS_ST01_OK;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st01_init(uint16_t apid)
{
    if (apid > 0x7FFU) {
        return PUS_ST01_ERR_PARAM;
    }

    st01_apid = apid;
    st01_init_done = 1U;
    return PUS_ST01_OK;
}

int32_t pus_st01_accept_ok(const ccsds_packet_t *tc_pkt)
{
    return build_verif_report(PUS_ST01_ACCEPT_OK, tc_pkt, 0U, 0U);
}

int32_t pus_st01_accept_fail(const ccsds_packet_t *tc_pkt, uint16_t err_code)
{
    return build_verif_report(PUS_ST01_ACCEPT_FAIL, tc_pkt, err_code, 1U);
}

int32_t pus_st01_start_ok(const ccsds_packet_t *tc_pkt)
{
    return build_verif_report(PUS_ST01_START_OK, tc_pkt, 0U, 0U);
}

int32_t pus_st01_exec_ok(const ccsds_packet_t *tc_pkt)
{
    return build_verif_report(PUS_ST01_EXEC_OK, tc_pkt, 0U, 0U);
}

int32_t pus_st01_exec_fail(const ccsds_packet_t *tc_pkt, uint16_t err_code)
{
    return build_verif_report(PUS_ST01_EXEC_FAIL, tc_pkt, err_code, 1U);
}
