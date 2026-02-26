/**
 * @file tm_framing.h
 * @brief CCSDS TM Transfer Frame (CCSDS 132.0-B-3) interface.
 *
 * Implements TM Transfer Frame assembly with Virtual Channel
 * multiplexing, idle frame insertion, and FECF (CRC-32) error detection.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef TM_FRAMING_H
#define TM_FRAMING_H

#include <stdint.h>

/** TM Transfer Frame sizes */
#define TM_FRAME_PRI_HDR_SIZE   6U      /**< Primary header              */
#define TM_FRAME_SEC_HDR_SIZE   4U      /**< Secondary header (optional) */
#define TM_FRAME_FECF_SIZE      2U      /**< Frame Error Control Field   */
#define TM_FRAME_MAX_DATA       1109U   /**< Max data field length       */
#define TM_FRAME_MAX_SIZE       (TM_FRAME_PRI_HDR_SIZE + TM_FRAME_SEC_HDR_SIZE + \
                                 TM_FRAME_MAX_DATA + TM_FRAME_FECF_SIZE)

/** Number of Virtual Channels */
#define TM_NUM_VC              8U

/** Frame Version Number */
#define TM_FRAME_VERSION       0U

/** Idle VC (virtual channel 7 by convention) */
#define TM_VC_IDLE             7U

/**
 * @brief TM Transfer Frame primary header.
 */
typedef struct {
    uint16_t tfvn_scid;         /**< TFVN(2) | SCID(10) | VCID(3) | OCF(1) */
    uint8_t  mc_fc;             /**< Master Channel Frame Counter           */
    uint8_t  vc_fc;             /**< Virtual Channel Frame Counter          */
    uint16_t fhp;               /**< Frame Data Field Status (FHP, etc.)    */
} tm_frame_hdr_t;

/**
 * @brief Complete TM Transfer Frame.
 */
typedef struct {
    tm_frame_hdr_t header;
    uint8_t        sec_hdr[TM_FRAME_SEC_HDR_SIZE]; /**< Secondary header   */
    uint8_t        data[TM_FRAME_MAX_DATA];         /**< Data field         */
    uint32_t       data_len;                        /**< Actual data length */
    uint16_t       fecf;                            /**< Frame Error Control*/
} tm_frame_t;

/** Return codes */
#define TM_OK               0
#define TM_ERR_PARAM        (-1)
#define TM_ERR_SIZE         (-2)
#define TM_ERR_CRC          (-3)

/**
 * @brief Initialize TM framing subsystem.
 * @param[in] scid Spacecraft ID (10-bit).
 * @return TM_OK on success.
 */
int32_t tm_framing_init(uint16_t scid);

/**
 * @brief Build a TM Transfer Frame.
 * @param[out] frame   Frame structure.
 * @param[in]  vcid    Virtual Channel ID (0-7).
 * @param[in]  data    Source data (space packets).
 * @param[in]  len     Data length.
 * @param[in]  fhp     First Header Pointer (0x7FF = no packet start).
 * @return TM_OK on success.
 */
int32_t tm_build_frame(tm_frame_t *frame, uint8_t vcid,
                        const uint8_t *data, uint32_t len,
                        uint16_t fhp);

/**
 * @brief Serialize TM frame to wire bytes.
 * @param[in]  frame  Frame structure.
 * @param[out] buf    Output buffer.
 * @param[in]  buflen Buffer size.
 * @param[out] outlen Serialized length.
 * @return TM_OK on success.
 */
int32_t tm_serialize_frame(const tm_frame_t *frame, uint8_t *buf,
                            uint32_t buflen, uint32_t *outlen);

/**
 * @brief Deserialize wire bytes to TM frame.
 * @param[in]  buf   Input buffer.
 * @param[in]  len   Buffer length.
 * @param[out] frame Frame structure.
 * @return TM_OK on success, TM_ERR_CRC on FECF mismatch.
 */
int32_t tm_deserialize_frame(const uint8_t *buf, uint32_t len,
                              tm_frame_t *frame);

/**
 * @brief Build an idle TM frame for VC fill.
 * @param[out] frame Frame structure.
 * @return TM_OK on success.
 */
int32_t tm_build_idle_frame(tm_frame_t *frame);

/**
 * @brief Get next Master Channel frame counter.
 * @return Frame counter value (0-255, wrapping).
 */
uint8_t tm_next_mc_counter(void);

/**
 * @brief Get next Virtual Channel frame counter for a VC.
 * @param[in] vcid Virtual channel ID.
 * @return Frame counter value (0-255, wrapping).
 */
uint8_t tm_next_vc_counter(uint8_t vcid);

#endif /* TM_FRAMING_H */
