/**
 * @file tc_framing.h
 * @brief CCSDS TC Transfer Frame and CLTU interface.
 *
 * Implements TC Transfer Frame (CCSDS 232.0-B-4) and
 * Communications Link Transmission Unit (CLTU) with BCH encoding
 * for uplink telecommand processing.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef TC_FRAMING_H
#define TC_FRAMING_H

#include <stdint.h>

/** TC Transfer Frame sizes */
#define TC_FRAME_PRI_HDR_SIZE   5U      /**< Primary header size         */
#define TC_FRAME_FECF_SIZE      2U      /**< Frame Error Control (CRC-16)*/
#define TC_FRAME_MAX_DATA       1019U   /**< Max data field length       */
#define TC_FRAME_MAX_SIZE       (TC_FRAME_PRI_HDR_SIZE + TC_FRAME_MAX_DATA + \
                                 TC_FRAME_FECF_SIZE)

/** CLTU constants */
#define CLTU_START_SEQ_LEN      2U      /**< Start sequence bytes        */
#define CLTU_CODEBLOCK_INFO     7U      /**< Info bytes per codeblock    */
#define CLTU_CODEBLOCK_SIZE     8U      /**< Total bytes per codeblock   */
#define CLTU_TAIL_SEQ_LEN       8U      /**< Tail sequence bytes         */
#define CLTU_MAX_SIZE           1280U   /**< Max CLTU size               */

/** TC Frame Version Number */
#define TC_FRAME_VERSION        0U

/** Return codes */
#define TC_OK               0
#define TC_ERR_PARAM        (-1)
#define TC_ERR_SIZE         (-2)
#define TC_ERR_CRC          (-3)
#define TC_ERR_BCH          (-4)

/**
 * @brief TC Transfer Frame primary header.
 */
typedef struct {
    uint16_t tfvn_flags;    /**< TFVN(2)|Bypass(1)|CtrlCmd(1)|Rsvd(2)|SCID(10)  */
    uint16_t vcid_len;      /**< VCID(6)|FrameLen(10)                            */
    uint8_t  seq;           /**< Frame Sequence Number (8 bits)                  */
} tc_frame_hdr_t;

/**
 * @brief TC Transfer Frame.
 */
typedef struct {
    tc_frame_hdr_t header;
    uint8_t        data[TC_FRAME_MAX_DATA];
    uint32_t       data_len;
    uint16_t       fecf;
} tc_frame_t;

/**
 * @brief Initialize TC framing subsystem.
 * @param[in] scid Spacecraft ID (10-bit).
 * @return TC_OK on success.
 */
int32_t tc_framing_init(uint16_t scid);

/**
 * @brief Build a TC Transfer Frame.
 * @param[out] frame  Frame structure.
 * @param[in]  vcid   Virtual Channel ID (0-63).
 * @param[in]  bypass Bypass flag (AD=0, BD=1).
 * @param[in]  data   TC data (space packets).
 * @param[in]  len    Data length.
 * @return TC_OK on success.
 */
int32_t tc_build_frame(tc_frame_t *frame, uint8_t vcid, uint8_t bypass,
                        const uint8_t *data, uint32_t len);

/**
 * @brief Serialize TC frame to wire bytes.
 * @param[in]  frame  Frame structure.
 * @param[out] buf    Output buffer.
 * @param[in]  buflen Buffer size.
 * @param[out] outlen Serialized length.
 * @return TC_OK on success.
 */
int32_t tc_serialize_frame(const tc_frame_t *frame, uint8_t *buf,
                            uint32_t buflen, uint32_t *outlen);

/**
 * @brief Deserialize and validate a TC Transfer Frame.
 * @param[in]  buf   Input buffer.
 * @param[in]  len   Buffer length.
 * @param[out] frame Frame structure.
 * @return TC_OK on success.
 */
int32_t tc_deserialize_frame(const uint8_t *buf, uint32_t len,
                              tc_frame_t *frame);

/**
 * @brief Encode TC frame data into CLTU with BCH codeblocks.
 * @param[in]  tc_data TC frame bytes.
 * @param[in]  tc_len  TC frame length.
 * @param[out] cltu    CLTU output buffer.
 * @param[in]  cltu_sz CLTU buffer size.
 * @param[out] cltu_len Actual CLTU length.
 * @return TC_OK on success.
 */
int32_t tc_encode_cltu(const uint8_t *tc_data, uint32_t tc_len,
                        uint8_t *cltu, uint32_t cltu_sz,
                        uint32_t *cltu_len);

/**
 * @brief Decode CLTU to TC frame data with BCH verification.
 * @param[in]  cltu    CLTU input buffer.
 * @param[in]  cltu_len CLTU length.
 * @param[out] tc_data  TC frame data output.
 * @param[in]  tc_sz    TC buffer size.
 * @param[out] tc_len   Actual TC frame length.
 * @return TC_OK on success, TC_ERR_BCH on parity error.
 */
int32_t tc_decode_cltu(const uint8_t *cltu, uint32_t cltu_len,
                        uint8_t *tc_data, uint32_t tc_sz,
                        uint32_t *tc_len);

#endif /* TC_FRAMING_H */
