/**
 * @file pus_st17.h
 * @brief PUS Service 17 — Test Connection (ECSS-E-ST-70-41C).
 *
 * "Are You Alive" test. TC(17,1) → TM(17,2).
 * Simplest PUS service for communication verification.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST17_H
#define PUS_ST17_H

#include <stdint.h>

/** PUS ST17 subtypes */
#define PUS_ST17_ARE_YOU_ALIVE  1U  /**< TC(17,1) Are you alive?   */
#define PUS_ST17_I_AM_ALIVE     2U  /**< TM(17,2) I am alive       */

/** Return codes */
#define PUS_ST17_OK         0
#define PUS_ST17_ERR_PARAM  (-1)

/**
 * @brief Initialize PUS Service 17.
 * @param[in] apid TM source APID.
 * @return PUS_ST17_OK on success.
 */
int32_t pus_st17_init(uint16_t apid);

/**
 * @brief Handle TC(17,1) — respond with TM(17,2).
 * @return PUS_ST17_OK on success.
 */
int32_t pus_st17_handle(void);

#endif /* PUS_ST17_H */
