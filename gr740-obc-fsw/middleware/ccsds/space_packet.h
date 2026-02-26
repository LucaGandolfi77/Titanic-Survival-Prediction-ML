/**
 * @file space_packet.h
 * @brief CCSDS Space Packet Protocol (CCSDS 133.0-B-2) interface.
 *
 * Implements CCSDS Space Packet primary header encoding/decoding,
 * packet assembly/disassembly, and CRC-16 error detection.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef SPACE_PACKET_H
#define SPACE_PACKET_H

#include <stdint.h>

/** CCSDS Space Packet primary header size in bytes */
#define CCSDS_PRI_HDR_SIZE      6U

/** Maximum packet data field length */
#define CCSDS_MAX_DATA_LEN      4089U

/** Maximum total packet size (header + data + CRC-16) */
#define CCSDS_MAX_PKT_SIZE      (CCSDS_PRI_HDR_SIZE + CCSDS_MAX_DATA_LEN + 2U)

/** Packet Version Number */
#define CCSDS_VERSION           0U

/** Packet Type */
#define CCSDS_TYPE_TM           0U  /**< Telemetry     */
#define CCSDS_TYPE_TC           1U  /**< Telecommand   */

/** Secondary Header Flag */
#define CCSDS_SHDR_PRESENT      1U
#define CCSDS_SHDR_ABSENT       0U

/** Sequence Flags */
#define CCSDS_SEQ_CONT          0U  /**< Continuation segment */
#define CCSDS_SEQ_FIRST         1U  /**< First segment        */
#define CCSDS_SEQ_LAST          2U  /**< Last segment         */
#define CCSDS_SEQ_UNSEG         3U  /**< Unsegmented          */

/** Sequence count mask (14 bits) */
#define CCSDS_SEQ_COUNT_MASK    0x3FFFU

/**
 * @brief CCSDS Space Packet primary header (6 bytes).
 */
typedef struct {
    uint16_t pkt_id;        /**< Version(3) | Type(1) | SHF(1) | APID(11) */
    uint16_t pkt_seq_ctrl;  /**< SeqFlags(2) | SeqCount(14)               */
    uint16_t pkt_data_len;  /**< Data Field Length - 1                     */
} ccsds_pri_hdr_t;

/**
 * @brief Complete CCSDS Space Packet structure.
 */
typedef struct {
    ccsds_pri_hdr_t header;                        /**< Primary header      */
    uint8_t         data[CCSDS_MAX_DATA_LEN];      /**< Data field          */
    uint16_t        crc;                           /**< CRC-16 CCITT       */
    uint32_t        data_len;                      /**< Actual data length  */
} ccsds_packet_t;

/** Return codes */
#define CCSDS_OK            0
#define CCSDS_ERR_PARAM     (-1)
#define CCSDS_ERR_SIZE      (-2)
#define CCSDS_ERR_CRC       (-3)

/**
 * @brief Initialize a CCSDS packet header.
 * @param[out] pkt       Packet structure.
 * @param[in]  type      CCSDS_TYPE_TM or CCSDS_TYPE_TC.
 * @param[in]  apid      Application Process Identifier (11-bit).
 * @param[in]  seq_flags Sequence flags (CCSDS_SEQ_*).
 * @param[in]  sec_hdr   Secondary header flag.
 * @return CCSDS_OK on success.
 */
int32_t ccsds_init_packet(ccsds_packet_t *pkt, uint8_t type,
                           uint16_t apid, uint8_t seq_flags,
                           uint8_t sec_hdr);

/**
 * @brief Set packet data.
 * @param[in,out] pkt  Packet structure.
 * @param[in]     data Data bytes.
 * @param[in]     len  Data length.
 * @return CCSDS_OK on success.
 */
int32_t ccsds_set_data(ccsds_packet_t *pkt, const uint8_t *data, uint32_t len);

/**
 * @brief Serialize packet to wire format (big-endian).
 * @param[in]  pkt    Packet structure.
 * @param[out] buf    Output buffer.
 * @param[in]  buflen Buffer size.
 * @param[out] outlen Actual serialized length.
 * @return CCSDS_OK on success.
 */
int32_t ccsds_serialize(const ccsds_packet_t *pkt, uint8_t *buf,
                         uint32_t buflen, uint32_t *outlen);

/**
 * @brief Deserialize wire-format buffer to packet structure.
 * @param[in]  buf    Input buffer.
 * @param[in]  len    Buffer length.
 * @param[out] pkt    Packet structure.
 * @return CCSDS_OK on success, CCSDS_ERR_CRC on CRC mismatch.
 */
int32_t ccsds_deserialize(const uint8_t *buf, uint32_t len,
                           ccsds_packet_t *pkt);

/**
 * @brief Compute CCSDS CRC-16 CCITT (polynomial 0x1021, init 0xFFFF).
 * @param[in] data Data buffer.
 * @param[in] len  Data length.
 * @return CRC-16 value.
 */
uint16_t ccsds_crc16(const uint8_t *data, uint32_t len);

/**
 * @brief Extract APID from packet.
 * @param[in] pkt Packet.
 * @return 11-bit APID.
 */
uint16_t ccsds_get_apid(const ccsds_packet_t *pkt);

/**
 * @brief Get packet type (TM/TC).
 * @param[in] pkt Packet.
 * @return CCSDS_TYPE_TM or CCSDS_TYPE_TC.
 */
uint8_t ccsds_get_type(const ccsds_packet_t *pkt);

/**
 * @brief Get/set sequence count for a given APID.
 * Manages per-APID sequence counters.
 * @param[in] apid APID.
 * @return Next sequence count (auto-increments, wraps at 16383).
 */
uint16_t ccsds_next_seq_count(uint16_t apid);

#endif /* SPACE_PACKET_H */
