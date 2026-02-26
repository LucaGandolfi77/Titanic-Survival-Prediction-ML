/**
 * @file tm_framing.c
 * @brief CCSDS TM Transfer Frame implementation.
 *
 * Compliant with CCSDS 132.0-B-3. Provides TM frame assembly/disassembly
 * with 8 Virtual Channels, FECF (CRC-16 CCITT), per-VC and master channel
 * frame counters, idle frame generation, and First Header Pointer tracking.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "tm_framing.h"
#include "space_packet.h"  /* For CRC-16 */

/* ── Module state ──────────────────────────────────────────────────────── */
static uint16_t tm_scid       = 0U;
static uint8_t  tm_mc_counter = 0U;
static uint8_t  tm_vc_counter[TM_NUM_VC];
static uint8_t  tm_initialized = 0U;

/* ── Idle fill pattern (CCSDS recommended: all 1s) ─────────────────────── */
#define IDLE_FILL_BYTE  0xFFU

/* ── First Header Pointer: no packet starts in this frame ──────────────── */
#define FHP_NO_PACKET   0x7FFU
#define FHP_IDLE        0x7FEU

/* ── Helper: big-endian encode ─────────────────────────────────────────── */
static inline void encode_be16(uint8_t *buf, uint16_t val)
{
    buf[0] = (uint8_t)((val >> 8U) & 0xFFU);
    buf[1] = (uint8_t)(val & 0xFFU);
}

static inline uint16_t decode_be16(const uint8_t *buf)
{
    return (uint16_t)(((uint16_t)buf[0] << 8U) | (uint16_t)buf[1]);
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t tm_framing_init(uint16_t scid)
{
    uint32_t i;

    if (scid > 0x3FFU) { /* 10-bit SCID */
        return TM_ERR_PARAM;
    }

    tm_scid = scid;
    tm_mc_counter = 0U;

    for (i = 0U; i < TM_NUM_VC; i++) {
        tm_vc_counter[i] = 0U;
    }

    tm_initialized = 1U;
    return TM_OK;
}

uint8_t tm_next_mc_counter(void)
{
    uint8_t val = tm_mc_counter;
    tm_mc_counter = (uint8_t)((tm_mc_counter + 1U) & 0xFFU);
    return val;
}

uint8_t tm_next_vc_counter(uint8_t vcid)
{
    uint8_t val;
    if (vcid >= TM_NUM_VC) {
        return 0U;
    }

    val = tm_vc_counter[vcid];
    tm_vc_counter[vcid] = (uint8_t)((tm_vc_counter[vcid] + 1U) & 0xFFU);
    return val;
}

int32_t tm_build_frame(tm_frame_t *frame, uint8_t vcid,
                        const uint8_t *data, uint32_t len,
                        uint16_t fhp)
{
    uint32_t i;

    if (frame == (tm_frame_t *)0) {
        return TM_ERR_PARAM;
    }
    if (vcid >= TM_NUM_VC) {
        return TM_ERR_PARAM;
    }
    if ((data == (const uint8_t *)0) && (len > 0U)) {
        return TM_ERR_PARAM;
    }
    if (len > TM_FRAME_MAX_DATA) {
        return TM_ERR_SIZE;
    }

    /*
     * Primary Header Word 0 (16 bits):
     * Bits 15-14: Transfer Frame Version Number (00)
     * Bits 13-4:  Spacecraft ID (10 bits)
     * Bits 3-1:   Virtual Channel ID (3 bits)
     * Bit 0:      Operational Control Field Flag (0 = no OCF)
     */
    frame->header.tfvn_scid = (uint16_t)(
        ((uint16_t)TM_FRAME_VERSION << 14U) |
        ((tm_scid & 0x3FFU) << 4U) |
        ((uint16_t)(vcid & 0x07U) << 1U) |
        0U  /* No OCF */
    );

    /* Frame counters */
    frame->header.mc_fc = tm_next_mc_counter();
    frame->header.vc_fc = tm_next_vc_counter(vcid);

    /*
     * Frame Data Field Status (16 bits):
     * Bit 15:    Secondary Header Flag (1 = present)
     * Bit 14:    Sync Flag (0 = forward-ordered)
     * Bit 13:    Packet Order Flag (0)
     * Bits 12-11: Segment Length ID (11 = unsegmented)
     * Bits 10-0:  First Header Pointer
     */
    frame->header.fhp = (uint16_t)(
        (1U << 15U) |       /* Secondary header present */
        (0U << 14U) |       /* Forward ordering         */
        (0U << 13U) |       /* No reordering            */
        (3U << 11U) |       /* Unsegmented              */
        (fhp & 0x7FFU)
    );

    /* Secondary header: 4-byte timestamp (CUC epoch offset — placeholder) */
    frame->sec_hdr[0] = 0U;
    frame->sec_hdr[1] = 0U;
    frame->sec_hdr[2] = 0U;
    frame->sec_hdr[3] = 0U;

    /* Copy data field */
    for (i = 0U; i < len; i++) {
        frame->data[i] = data[i];
    }

    /* Pad remainder with idle fill */
    for (i = len; i < TM_FRAME_MAX_DATA; i++) {
        frame->data[i] = IDLE_FILL_BYTE;
    }

    frame->data_len = len;
    frame->fecf = 0U; /* Computed during serialization */

    return TM_OK;
}

int32_t tm_serialize_frame(const tm_frame_t *frame, uint8_t *buf,
                            uint32_t buflen, uint32_t *outlen)
{
    uint32_t pos;
    uint32_t frame_size;
    uint32_t i;
    uint16_t fecf;

    if ((frame == (const tm_frame_t *)0) || (buf == (uint8_t *)0) ||
        (outlen == (uint32_t *)0)) {
        return TM_ERR_PARAM;
    }

    /* Fixed frame size: hdr(6) + sec_hdr(4) + data(MAX) + FECF(2) */
    frame_size = TM_FRAME_PRI_HDR_SIZE + TM_FRAME_SEC_HDR_SIZE +
                 TM_FRAME_MAX_DATA + TM_FRAME_FECF_SIZE;

    if (buflen < frame_size) {
        return TM_ERR_SIZE;
    }

    pos = 0U;

    /* Primary header */
    encode_be16(&buf[pos], frame->header.tfvn_scid);
    pos += 2U;
    buf[pos] = frame->header.mc_fc;
    pos += 1U;
    buf[pos] = frame->header.vc_fc;
    pos += 1U;
    encode_be16(&buf[pos], frame->header.fhp);
    pos += 2U;

    /* Secondary header */
    for (i = 0U; i < TM_FRAME_SEC_HDR_SIZE; i++) {
        buf[pos] = frame->sec_hdr[i];
        pos += 1U;
    }

    /* Data field (always full frame length) */
    for (i = 0U; i < TM_FRAME_MAX_DATA; i++) {
        buf[pos] = frame->data[i];
        pos += 1U;
    }

    /* Compute FECF (CRC-16 CCITT over entire frame before FECF) */
    fecf = ccsds_crc16(buf, pos);
    encode_be16(&buf[pos], fecf);
    pos += 2U;

    *outlen = pos;
    return TM_OK;
}

int32_t tm_deserialize_frame(const uint8_t *buf, uint32_t len,
                              tm_frame_t *frame)
{
    uint32_t frame_size;
    uint32_t pos;
    uint32_t i;
    uint16_t fecf_recv;
    uint16_t fecf_calc;

    if ((buf == (const uint8_t *)0) || (frame == (tm_frame_t *)0)) {
        return TM_ERR_PARAM;
    }

    frame_size = TM_FRAME_PRI_HDR_SIZE + TM_FRAME_SEC_HDR_SIZE +
                 TM_FRAME_MAX_DATA + TM_FRAME_FECF_SIZE;

    if (len < frame_size) {
        return TM_ERR_SIZE;
    }

    pos = 0U;

    /* Primary header */
    frame->header.tfvn_scid = decode_be16(&buf[pos]);
    pos += 2U;
    frame->header.mc_fc = buf[pos];
    pos += 1U;
    frame->header.vc_fc = buf[pos];
    pos += 1U;
    frame->header.fhp = decode_be16(&buf[pos]);
    pos += 2U;

    /* Secondary header */
    for (i = 0U; i < TM_FRAME_SEC_HDR_SIZE; i++) {
        frame->sec_hdr[i] = buf[pos];
        pos += 1U;
    }

    /* Data field */
    for (i = 0U; i < TM_FRAME_MAX_DATA; i++) {
        frame->data[i] = buf[pos];
        pos += 1U;
    }
    frame->data_len = TM_FRAME_MAX_DATA;

    /* Verify FECF */
    fecf_recv = decode_be16(&buf[pos]);
    fecf_calc = ccsds_crc16(buf, pos);

    if (fecf_recv != fecf_calc) {
        return TM_ERR_CRC;
    }

    frame->fecf = fecf_recv;
    return TM_OK;
}

int32_t tm_build_idle_frame(tm_frame_t *frame)
{
    uint32_t i;
    uint8_t  idle_data[TM_FRAME_MAX_DATA];

    if (frame == (tm_frame_t *)0) {
        return TM_ERR_PARAM;
    }

    for (i = 0U; i < TM_FRAME_MAX_DATA; i++) {
        idle_data[i] = IDLE_FILL_BYTE;
    }

    return tm_build_frame(frame, TM_VC_IDLE, idle_data,
                           TM_FRAME_MAX_DATA, FHP_IDLE);
}
