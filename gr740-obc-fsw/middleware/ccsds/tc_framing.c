/**
 * @file tc_framing.c
 * @brief CCSDS TC Transfer Frame and CLTU implementation.
 *
 * Compliant with CCSDS 232.0-B-4 (TC Space Data Link Protocol)
 * and CCSDS 231.0-B-4 (TC Synchronization and Channel Coding).
 * Implements TC frame assembly/disassembly, CRC-16 FECF, and
 * CLTU encoding with (63,56) BCH shortened codeblocks.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "tc_framing.h"
#include "space_packet.h"  /* For CRC-16 */

/* ── Module state ──────────────────────────────────────────────────────── */
static uint16_t tc_scid = 0U;
static uint8_t  tc_seq_counter[64]; /* Per-VCID sequence counters */
static uint8_t  tc_initialized = 0U;

/* ── CLTU Start and Tail sequences ─────────────────────────────────────── */
static const uint8_t cltu_start_seq[CLTU_START_SEQ_LEN] = { 0xEBU, 0x90U };
static const uint8_t cltu_tail_seq[CLTU_TAIL_SEQ_LEN]   = {
    0xC5U, 0xC5U, 0xC5U, 0xC5U, 0xC5U, 0xC5U, 0xC5U, 0x79U
};

/* ── BCH parity computation ────────────────────────────────────────────── */

/**
 * @brief Compute BCH parity byte for a 7-byte information block.
 *
 * Uses the generator polynomial for the CCSDS (63,56) shortened BCH code.
 * The polynomial is x^7 + x^6 + x^2 + 1 = 0xC5 (or 0x45 without MSB).
 *
 * @param[in] data  7 bytes of information.
 * @return Parity byte including filler bit.
 */
static uint8_t bch_parity(const uint8_t *data)
{
    uint8_t sr = 0U;  /* Shift register */
    uint32_t i;
    uint32_t j;
    uint8_t  feedback;
    uint8_t  byte_val;

    for (i = 0U; i < CLTU_CODEBLOCK_INFO; i++) {
        byte_val = data[i];
        for (j = 0U; j < 8U; j++) {
            feedback = (uint8_t)(((sr >> 6U) ^ (byte_val >> 7U)) & 0x01U);
            sr = (uint8_t)((sr << 1U) & 0x7FU);
            if (feedback != 0U) {
                sr ^= 0x45U; /* x^6 + x^2 + 1 */
            }
            byte_val = (uint8_t)(byte_val << 1U);
        }
    }

    /* Complement and set filler bit (bit 0) */
    sr = (uint8_t)(~sr & 0xFEU);  /* Bits 7..1 = complemented parity, bit 0 = 0 (filler) */

    return sr;
}

/* ── Helper: big-endian ────────────────────────────────────────────────── */
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

int32_t tc_framing_init(uint16_t scid)
{
    uint32_t i;

    if (scid > 0x3FFU) {
        return TC_ERR_PARAM;
    }

    tc_scid = scid;

    for (i = 0U; i < 64U; i++) {
        tc_seq_counter[i] = 0U;
    }

    tc_initialized = 1U;
    return TC_OK;
}

int32_t tc_build_frame(tc_frame_t *frame, uint8_t vcid, uint8_t bypass,
                        const uint8_t *data, uint32_t len)
{
    uint32_t frame_len;
    uint32_t i;

    if (frame == (tc_frame_t *)0) {
        return TC_ERR_PARAM;
    }
    if (vcid > 63U) {
        return TC_ERR_PARAM;
    }
    if ((data == (const uint8_t *)0) && (len > 0U)) {
        return TC_ERR_PARAM;
    }
    if (len > TC_FRAME_MAX_DATA) {
        return TC_ERR_SIZE;
    }

    /*
     * Primary Header Word 0 (16 bits):
     * Bits 15-14: TFVN (00)
     * Bit 13:     Bypass Flag (0=AD, 1=BD)
     * Bit 12:     Control Command Flag (0=data, 1=control)
     * Bits 11-10: Reserved (00)
     * Bits 9-0:   Spacecraft ID
     */
    frame->header.tfvn_flags = (uint16_t)(
        ((uint16_t)TC_FRAME_VERSION << 14U) |
        ((uint16_t)(bypass & 1U) << 13U) |
        (0U << 12U) |  /* Data frame */
        (tc_scid & 0x3FFU)
    );

    /*
     * Primary Header Word 1 (16 bits):
     * Bits 15-10: Virtual Channel ID (6 bits)
     * Bits 9-0:   Frame Length - 1
     */
    frame_len = TC_FRAME_PRI_HDR_SIZE + len + TC_FRAME_FECF_SIZE;
    frame->header.vcid_len = (uint16_t)(
        ((uint16_t)(vcid & 0x3FU) << 10U) |
        ((frame_len - 1U) & 0x3FFU)
    );

    /* Sequence number */
    frame->header.seq = tc_seq_counter[vcid];
    tc_seq_counter[vcid] = (uint8_t)((tc_seq_counter[vcid] + 1U) & 0xFFU);

    /* Copy data */
    for (i = 0U; i < len; i++) {
        frame->data[i] = data[i];
    }
    frame->data_len = len;
    frame->fecf = 0U;

    return TC_OK;
}

int32_t tc_serialize_frame(const tc_frame_t *frame, uint8_t *buf,
                            uint32_t buflen, uint32_t *outlen)
{
    uint32_t frame_len;
    uint32_t pos;
    uint32_t i;
    uint16_t fecf;

    if ((frame == (const tc_frame_t *)0) || (buf == (uint8_t *)0) ||
        (outlen == (uint32_t *)0)) {
        return TC_ERR_PARAM;
    }

    frame_len = TC_FRAME_PRI_HDR_SIZE + frame->data_len + TC_FRAME_FECF_SIZE;

    if (buflen < frame_len) {
        return TC_ERR_SIZE;
    }

    pos = 0U;

    /* Primary header */
    encode_be16(&buf[pos], frame->header.tfvn_flags);
    pos += 2U;
    encode_be16(&buf[pos], frame->header.vcid_len);
    pos += 2U;
    buf[pos] = frame->header.seq;
    pos += 1U;

    /* Data field */
    for (i = 0U; i < frame->data_len; i++) {
        buf[pos] = frame->data[i];
        pos += 1U;
    }

    /* Compute and append FECF (CRC-16 CCITT) */
    fecf = ccsds_crc16(buf, pos);
    encode_be16(&buf[pos], fecf);
    pos += 2U;

    *outlen = pos;
    return TC_OK;
}

int32_t tc_deserialize_frame(const uint8_t *buf, uint32_t len,
                              tc_frame_t *frame)
{
    uint32_t frame_len;
    uint32_t pos;
    uint32_t i;
    uint16_t fecf_recv;
    uint16_t fecf_calc;

    if ((buf == (const uint8_t *)0) || (frame == (tc_frame_t *)0)) {
        return TC_ERR_PARAM;
    }

    if (len < (TC_FRAME_PRI_HDR_SIZE + TC_FRAME_FECF_SIZE)) {
        return TC_ERR_SIZE;
    }

    pos = 0U;

    /* Decode primary header */
    frame->header.tfvn_flags = decode_be16(&buf[pos]);
    pos += 2U;
    frame->header.vcid_len = decode_be16(&buf[pos]);
    pos += 2U;
    frame->header.seq = buf[pos];
    pos += 1U;

    /* Extract frame length from header */
    frame_len = (uint32_t)(frame->header.vcid_len & 0x3FFU) + 1U;

    if (len < frame_len) {
        return TC_ERR_SIZE;
    }

    /* Data length = frame_len - header - FECF */
    frame->data_len = frame_len - TC_FRAME_PRI_HDR_SIZE - TC_FRAME_FECF_SIZE;

    if (frame->data_len > TC_FRAME_MAX_DATA) {
        return TC_ERR_SIZE;
    }

    /* Copy data field */
    for (i = 0U; i < frame->data_len; i++) {
        frame->data[i] = buf[pos];
        pos += 1U;
    }

    /* Verify FECF */
    fecf_recv = decode_be16(&buf[pos]);
    fecf_calc = ccsds_crc16(buf, pos);

    if (fecf_recv != fecf_calc) {
        return TC_ERR_CRC;
    }

    frame->fecf = fecf_recv;
    return TC_OK;
}

int32_t tc_encode_cltu(const uint8_t *tc_data, uint32_t tc_len,
                        uint8_t *cltu, uint32_t cltu_sz,
                        uint32_t *cltu_len)
{
    uint32_t num_blocks;
    uint32_t pos_in;
    uint32_t pos_out;
    uint32_t i;
    uint32_t b;
    uint8_t  block[CLTU_CODEBLOCK_INFO];
    uint8_t  parity;
    uint32_t required_size;

    if ((tc_data == (const uint8_t *)0) || (cltu == (uint8_t *)0) ||
        (cltu_len == (uint32_t *)0)) {
        return TC_ERR_PARAM;
    }
    if (tc_len == 0U) {
        return TC_ERR_PARAM;
    }

    /* Calculate number of codeblocks needed */
    num_blocks = (tc_len + CLTU_CODEBLOCK_INFO - 1U) / CLTU_CODEBLOCK_INFO;

    /* Total CLTU size: start + blocks*8 + tail */
    required_size = CLTU_START_SEQ_LEN +
                    (num_blocks * CLTU_CODEBLOCK_SIZE) +
                    CLTU_TAIL_SEQ_LEN;

    if (cltu_sz < required_size) {
        return TC_ERR_SIZE;
    }

    pos_out = 0U;

    /* Start sequence */
    for (i = 0U; i < CLTU_START_SEQ_LEN; i++) {
        cltu[pos_out] = cltu_start_seq[i];
        pos_out++;
    }

    /* Encode codeblocks */
    pos_in = 0U;
    for (b = 0U; b < num_blocks; b++) {
        /* Fill 7-byte info block, pad with 0x55 (alternating fill) */
        for (i = 0U; i < CLTU_CODEBLOCK_INFO; i++) {
            if (pos_in < tc_len) {
                block[i] = tc_data[pos_in];
                pos_in++;
            } else {
                block[i] = 0x55U; /* Fill pattern */
            }
        }

        /* Write info bytes */
        for (i = 0U; i < CLTU_CODEBLOCK_INFO; i++) {
            cltu[pos_out] = block[i];
            pos_out++;
        }

        /* Compute and write BCH parity */
        parity = bch_parity(block);
        cltu[pos_out] = parity;
        pos_out++;
    }

    /* Tail sequence */
    for (i = 0U; i < CLTU_TAIL_SEQ_LEN; i++) {
        cltu[pos_out] = cltu_tail_seq[i];
        pos_out++;
    }

    *cltu_len = pos_out;
    return TC_OK;
}

int32_t tc_decode_cltu(const uint8_t *cltu, uint32_t cltu_len,
                        uint8_t *tc_data, uint32_t tc_sz,
                        uint32_t *tc_len)
{
    uint32_t pos;
    uint32_t out_pos;
    uint32_t i;
    uint8_t  block[CLTU_CODEBLOCK_INFO];
    uint8_t  recv_parity;
    uint8_t  calc_parity;
    uint8_t  is_tail;

    if ((cltu == (const uint8_t *)0) || (tc_data == (uint8_t *)0) ||
        (tc_len == (uint32_t *)0)) {
        return TC_ERR_PARAM;
    }

    /* Verify minimum length: start(2) + 1 block(8) + tail(8) */
    if (cltu_len < (CLTU_START_SEQ_LEN + CLTU_CODEBLOCK_SIZE + CLTU_TAIL_SEQ_LEN)) {
        return TC_ERR_SIZE;
    }

    /* Verify start sequence */
    if ((cltu[0] != cltu_start_seq[0]) || (cltu[1] != cltu_start_seq[1])) {
        return TC_ERR_PARAM;
    }

    pos = CLTU_START_SEQ_LEN;
    out_pos = 0U;

    /* Process codeblocks until tail sequence */
    while ((pos + CLTU_CODEBLOCK_SIZE) <= cltu_len) {
        /* Check if this is the tail sequence */
        is_tail = 1U;
        for (i = 0U; i < CLTU_TAIL_SEQ_LEN; i++) {
            if ((pos + i) < cltu_len) {
                if (cltu[pos + i] != cltu_tail_seq[i]) {
                    is_tail = 0U;
                    break;
                }
            } else {
                is_tail = 0U;
                break;
            }
        }

        if (is_tail != 0U) {
            break; /* Reached tail */
        }

        /* Extract info bytes */
        for (i = 0U; i < CLTU_CODEBLOCK_INFO; i++) {
            block[i] = cltu[pos + i];
        }
        recv_parity = cltu[pos + CLTU_CODEBLOCK_INFO];

        /* Verify BCH parity */
        calc_parity = bch_parity(block);
        if (recv_parity != calc_parity) {
            return TC_ERR_BCH;
        }

        /* Copy info bytes to output */
        for (i = 0U; i < CLTU_CODEBLOCK_INFO; i++) {
            if (out_pos < tc_sz) {
                tc_data[out_pos] = block[i];
                out_pos++;
            } else {
                return TC_ERR_SIZE;
            }
        }

        pos += CLTU_CODEBLOCK_SIZE;
    }

    *tc_len = out_pos;
    return TC_OK;
}
