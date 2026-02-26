/**
 * @file space_packet.c
 * @brief CCSDS Space Packet Protocol implementation.
 *
 * Compliant with CCSDS 133.0-B-2 (Space Packet Protocol).
 * Provides packet assembly, disassembly, serialization, and
 * CRC-16 CCITT error detection for TM and TC packets.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "space_packet.h"

/* ── Per-APID sequence counters ────────────────────────────────────────── */
#define MAX_APIDS   64U

static uint16_t seq_counters[MAX_APIDS];

/* ── CRC-16 CCITT Table (polynomial 0x1021, init 0xFFFF) ───────────────── */
static const uint16_t crc16_table[256] = {
    0x0000U, 0x1021U, 0x2042U, 0x3063U, 0x4084U, 0x50A5U, 0x60C6U, 0x70E7U,
    0x8108U, 0x9129U, 0xA14AU, 0xB16BU, 0xC18CU, 0xD1ADU, 0xE1CEU, 0xF1EFU,
    0x1231U, 0x0210U, 0x3273U, 0x2252U, 0x52B5U, 0x4294U, 0x72F7U, 0x62D6U,
    0x9339U, 0x8318U, 0xB37BU, 0xA35AU, 0xD3BDU, 0xC39CU, 0xF3FFU, 0xE3DEU,
    0x2462U, 0x3443U, 0x0420U, 0x1401U, 0x64E6U, 0x74C7U, 0x44A4U, 0x5485U,
    0xA56AU, 0xB54BU, 0x8528U, 0x9509U, 0xE5EEU, 0xF5CFU, 0xC5ACU, 0xD58DU,
    0x3653U, 0x2672U, 0x1611U, 0x0630U, 0x76D7U, 0x66F6U, 0x5695U, 0x46B4U,
    0xB75BU, 0xA77AU, 0x9719U, 0x8738U, 0xF7DFU, 0xE7FEU, 0xD79DU, 0xC7BCU,
    0x4864U, 0x5845U, 0x6826U, 0x7807U, 0x08E0U, 0x18C1U, 0x28A2U, 0x3883U,
    0xC96CU, 0xD94DU, 0xE92EU, 0xF90FU, 0x89E8U, 0x99C9U, 0xA9AAU, 0xB98BU,
    0x5A55U, 0x4A74U, 0x7A17U, 0x6A36U, 0x1AD1U, 0x0AF0U, 0x3A93U, 0x2AB2U,
    0xDB5DU, 0xCB7CU, 0xFB1FU, 0xEB3EU, 0x9BD9U, 0x8BF8U, 0xBB9BU, 0xAB9AU,
    0x6CA6U, 0x7C87U, 0x4CE4U, 0x5CC5U, 0x2C22U, 0x3C03U, 0x0C60U, 0x1C41U,
    0xEDAEU, 0xFD8FU, 0xCDECU, 0xDDCDU, 0xAD2AU, 0xBD0BU, 0x8D68U, 0x9D49U,
    0x7E97U, 0x6EB6U, 0x5ED5U, 0x4EF4U, 0x3E13U, 0x2E32U, 0x1E51U, 0x0E70U,
    0xFF9FU, 0xEFBEU, 0xDFDDU, 0xCFFCU, 0xBF1BU, 0xAF3AU, 0x9F59U, 0x8F78U,
    0x9188U, 0x81A9U, 0xB1CAU, 0xA1EBU, 0xD10CU, 0xC12DU, 0xF14EU, 0xE16FU,
    0x1080U, 0x00A1U, 0x30C2U, 0x20E3U, 0x5004U, 0x4025U, 0x7046U, 0x6067U,
    0x83B9U, 0x9398U, 0xA3FBU, 0xB3DAU, 0xC33DU, 0xD31CU, 0xE37FU, 0xF35EU,
    0x02B1U, 0x1290U, 0x22F3U, 0x32D2U, 0x4235U, 0x5214U, 0x6277U, 0x7256U,
    0xB5EAU, 0xA5CBU, 0x95A8U, 0x8589U, 0xF56EU, 0xE54FU, 0xD52CU, 0xC50DU,
    0x34E2U, 0x24C3U, 0x14A0U, 0x0481U, 0x7466U, 0x6447U, 0x5424U, 0x4405U,
    0xA7DBU, 0xB7FAU, 0x8799U, 0x97B8U, 0xE75FU, 0xF77EU, 0xC71DU, 0xD73CU,
    0x26D3U, 0x36F2U, 0x0691U, 0x16B0U, 0x6657U, 0x7676U, 0x4615U, 0x5634U,
    0xD8ECU, 0xC8CDU, 0xF8AEU, 0xE88FU, 0x9868U, 0x8849U, 0xB82AU, 0xA80BU,
    0x59E4U, 0x49C5U, 0x79A6U, 0x6987U, 0x1960U, 0x0941U, 0x3922U, 0x2903U,
    0xCADDU, 0xDAFCU, 0xEA9FU, 0xFABEU, 0x8A59U, 0x9A78U, 0xAA1BU, 0xBA3AU,
    0x4BD5U, 0x5BF4U, 0x6B97U, 0x7BB6U, 0x0B51U, 0x1B70U, 0x2B13U, 0x3B32U,
    0x0CE8U, 0x1CC9U, 0x2CAAU, 0x3C8BU, 0x4C6CU, 0x5C4DU, 0x6C2EU, 0x7C0FU,
    0x8DE0U, 0x9DC1U, 0xADA2U, 0xBD83U, 0xCD64U, 0xDD45U, 0xED26U, 0xFD07U,
    0x1EF9U, 0x0ED8U, 0x3EBBU, 0x2E9AU, 0x5E7DU, 0x4E5CU, 0x7E3FU, 0x6E1EU,
    0x9FF1U, 0x8FD0U, 0xBFB3U, 0xAF92U, 0xDF75U, 0xCF54U, 0xFF37U, 0xEF16U
};

/* ── Helper: big-endian encode/decode ──────────────────────────────────── */
static inline void encode_u16(uint8_t *buf, uint16_t val)
{
    buf[0] = (uint8_t)((val >> 8U) & 0xFFU);
    buf[1] = (uint8_t)(val & 0xFFU);
}

static inline uint16_t decode_u16(const uint8_t *buf)
{
    return (uint16_t)(((uint16_t)buf[0] << 8U) | (uint16_t)buf[1]);
}

/* ── Public API ────────────────────────────────────────────────────────── */

uint16_t ccsds_crc16(const uint8_t *data, uint32_t len)
{
    uint16_t crc = 0xFFFFU;
    uint32_t i;
    uint8_t  idx;

    if (data == (const uint8_t *)0) {
        return 0xFFFFU;
    }

    for (i = 0U; i < len; i++) {
        idx = (uint8_t)((crc >> 8U) ^ data[i]);
        crc = (uint16_t)((crc << 8U) ^ crc16_table[idx]);
    }

    return crc;
}

int32_t ccsds_init_packet(ccsds_packet_t *pkt, uint8_t type,
                           uint16_t apid, uint8_t seq_flags,
                           uint8_t sec_hdr)
{
    if (pkt == (ccsds_packet_t *)0) {
        return CCSDS_ERR_PARAM;
    }
    if (apid > 0x7FFU) {
        return CCSDS_ERR_PARAM;
    }
    if (type > 1U) {
        return CCSDS_ERR_PARAM;
    }

    /* Construct Packet ID:
     * Bits 15-13: Version (000)
     * Bit 12:     Type (0=TM, 1=TC)
     * Bit 11:     Secondary Header Flag
     * Bits 10-0:  APID
     */
    pkt->header.pkt_id = (uint16_t)(
        ((uint16_t)CCSDS_VERSION << 13U) |
        ((uint16_t)type << 12U) |
        ((uint16_t)(sec_hdr & 1U) << 11U) |
        (apid & 0x7FFU)
    );

    /* Construct Packet Sequence Control:
     * Bits 15-14: Sequence Flags
     * Bits 13-0:  Sequence Count (set to 0, updated on serialize)
     */
    pkt->header.pkt_seq_ctrl = (uint16_t)((uint16_t)(seq_flags & 3U) << 14U);

    pkt->header.pkt_data_len = 0U;
    pkt->data_len = 0U;
    pkt->crc = 0U;

    return CCSDS_OK;
}

int32_t ccsds_set_data(ccsds_packet_t *pkt, const uint8_t *data, uint32_t len)
{
    uint32_t i;

    if (pkt == (ccsds_packet_t *)0) {
        return CCSDS_ERR_PARAM;
    }
    if (data == (const uint8_t *)0) {
        return CCSDS_ERR_PARAM;
    }
    if (len > CCSDS_MAX_DATA_LEN) {
        return CCSDS_ERR_SIZE;
    }

    for (i = 0U; i < len; i++) {
        pkt->data[i] = data[i];
    }
    pkt->data_len = len;

    /*
     * Packet Data Length field = (number of octets in data field) - 1
     * Data field = data + CRC(2 bytes)
     */
    pkt->header.pkt_data_len = (uint16_t)(len + 2U - 1U);

    return CCSDS_OK;
}

int32_t ccsds_serialize(const ccsds_packet_t *pkt, uint8_t *buf,
                         uint32_t buflen, uint32_t *outlen)
{
    uint32_t total_len;
    uint32_t i;
    uint16_t crc;

    if ((pkt == (const ccsds_packet_t *)0) || (buf == (uint8_t *)0) ||
        (outlen == (uint32_t *)0)) {
        return CCSDS_ERR_PARAM;
    }

    /* Total = header(6) + data + CRC(2) */
    total_len = CCSDS_PRI_HDR_SIZE + pkt->data_len + 2U;

    if (buflen < total_len) {
        return CCSDS_ERR_SIZE;
    }

    /* Encode primary header (big-endian) */
    encode_u16(&buf[0], pkt->header.pkt_id);
    encode_u16(&buf[2], pkt->header.pkt_seq_ctrl);
    encode_u16(&buf[4], pkt->header.pkt_data_len);

    /* Copy data field */
    for (i = 0U; i < pkt->data_len; i++) {
        buf[CCSDS_PRI_HDR_SIZE + i] = pkt->data[i];
    }

    /* Compute CRC-16 over header + data */
    crc = ccsds_crc16(buf, CCSDS_PRI_HDR_SIZE + pkt->data_len);

    /* Append CRC (big-endian) */
    encode_u16(&buf[CCSDS_PRI_HDR_SIZE + pkt->data_len], crc);

    *outlen = total_len;
    return CCSDS_OK;
}

int32_t ccsds_deserialize(const uint8_t *buf, uint32_t len,
                           ccsds_packet_t *pkt)
{
    uint16_t crc_received;
    uint16_t crc_computed;
    uint32_t data_field_len;
    uint32_t i;

    if ((buf == (const uint8_t *)0) || (pkt == (ccsds_packet_t *)0)) {
        return CCSDS_ERR_PARAM;
    }

    if (len < (CCSDS_PRI_HDR_SIZE + 2U)) {
        return CCSDS_ERR_SIZE;
    }

    /* Decode primary header */
    pkt->header.pkt_id       = decode_u16(&buf[0]);
    pkt->header.pkt_seq_ctrl = decode_u16(&buf[2]);
    pkt->header.pkt_data_len = decode_u16(&buf[4]);

    /* Data field length = pkt_data_len + 1 (includes CRC) */
    data_field_len = (uint32_t)pkt->header.pkt_data_len + 1U;

    /* Verify we have enough bytes */
    if (len < (CCSDS_PRI_HDR_SIZE + data_field_len)) {
        return CCSDS_ERR_SIZE;
    }

    /* Data length = data field - CRC(2) */
    pkt->data_len = data_field_len - 2U;

    if (pkt->data_len > CCSDS_MAX_DATA_LEN) {
        return CCSDS_ERR_SIZE;
    }

    /* Copy data */
    for (i = 0U; i < pkt->data_len; i++) {
        pkt->data[i] = buf[CCSDS_PRI_HDR_SIZE + i];
    }

    /* Extract received CRC */
    crc_received = decode_u16(&buf[CCSDS_PRI_HDR_SIZE + pkt->data_len]);

    /* Compute CRC over header + data */
    crc_computed = ccsds_crc16(buf, CCSDS_PRI_HDR_SIZE + pkt->data_len);

    if (crc_received != crc_computed) {
        return CCSDS_ERR_CRC;
    }

    pkt->crc = crc_received;
    return CCSDS_OK;
}

uint16_t ccsds_get_apid(const ccsds_packet_t *pkt)
{
    if (pkt == (const ccsds_packet_t *)0) {
        return 0x7FFU; /* Idle APID */
    }
    return pkt->header.pkt_id & 0x7FFU;
}

uint8_t ccsds_get_type(const ccsds_packet_t *pkt)
{
    if (pkt == (const ccsds_packet_t *)0) {
        return CCSDS_TYPE_TM;
    }
    return (uint8_t)((pkt->header.pkt_id >> 12U) & 1U);
}

uint16_t ccsds_next_seq_count(uint16_t apid)
{
    uint16_t idx;
    uint16_t count;

    idx = apid & (MAX_APIDS - 1U); /* Simple hash */
    count = seq_counters[idx];
    seq_counters[idx] = (uint16_t)((count + 1U) & CCSDS_SEQ_COUNT_MASK);

    return count;
}
