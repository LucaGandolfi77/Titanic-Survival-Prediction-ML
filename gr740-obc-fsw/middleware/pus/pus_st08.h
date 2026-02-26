/**
 * @file pus_st08.h
 * @brief PUS Service 8 — Function Management (ECSS-E-ST-70-41C).
 *
 * Allows ground to invoke predefined on-board functions via TC(8,1).
 * Execution result is reported via PUS Service 1 (verification).
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST08_H
#define PUS_ST08_H

#include <stdint.h>

/** PUS ST08 subtypes */
#define PUS_ST08_EXEC_FUNC  1U  /**< TC(8,1) Perform function */

/** Max registered functions */
#define PUS_ST08_MAX_FUNCS  32U

/** Max function argument size */
#define PUS_ST08_MAX_ARG    64U

/** Return codes */
#define PUS_ST08_OK             0
#define PUS_ST08_ERR_PARAM      (-1)
#define PUS_ST08_ERR_FULL       (-2)
#define PUS_ST08_ERR_NOT_FOUND  (-3)
#define PUS_ST08_ERR_EXEC       (-4)

/** Function handler callback */
typedef int32_t (*pus_func_handler_t)(const uint8_t *args, uint32_t arg_len);

/**
 * @brief Initialize PUS Service 8.
 * @return PUS_ST08_OK on success.
 */
int32_t pus_st08_init(void);

/**
 * @brief Register a function.
 * @param[in] func_id  Function ID.
 * @param[in] handler  Function handler.
 * @return PUS_ST08_OK on success.
 */
int32_t pus_st08_register(uint16_t func_id, pus_func_handler_t handler);

/**
 * @brief Execute a function by ID.
 * @param[in] func_id Function ID.
 * @param[in] args    Argument data (may be NULL).
 * @param[in] arg_len Argument length.
 * @return PUS_ST08_OK on success.
 */
int32_t pus_st08_execute(uint16_t func_id, const uint8_t *args, uint32_t arg_len);

/**
 * @brief Process a TC(8,1) packet data field.
 * @param[in] data Data field (after PUS sec hdr): func_id(2) + args.
 * @param[in] len  Data length.
 * @return PUS_ST08_OK on success.
 */
int32_t pus_st08_process(const uint8_t *data, uint32_t len);

#endif /* PUS_ST08_H */
