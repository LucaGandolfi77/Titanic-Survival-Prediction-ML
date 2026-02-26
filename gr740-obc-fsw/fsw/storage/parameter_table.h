/**
 * @file parameter_table.h
 * @brief Parameter Table — persistent configuration store.
 *
 * Stores mission-configurable parameters in MRAM with CRC protection.
 * Parameters are identified by a 16-bit ID and have a fixed max size.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PARAMETER_TABLE_H
#define PARAMETER_TABLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define PARAM_OK            0
#define PARAM_ERR_PARAM    -1
#define PARAM_ERR_NOT_FOUND -2
#define PARAM_ERR_FULL     -3
#define PARAM_ERR_NVM      -4

/* ── Configuration ─────────────────────────────────────────────────────── */
#define PARAM_MAX_ENTRIES    128U    /**< Max parameters                 */
#define PARAM_MAX_VALUE_SIZE  64U    /**< Max bytes per parameter value  */

/* ── Parameter type ────────────────────────────────────────────────────── */
typedef enum {
    PARAM_TYPE_UINT8   = 0,
    PARAM_TYPE_UINT16  = 1,
    PARAM_TYPE_UINT32  = 2,
    PARAM_TYPE_INT32   = 3,
    PARAM_TYPE_FLOAT   = 4,
    PARAM_TYPE_BLOB    = 5
} param_type_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the parameter table.
 * @return PARAM_OK on success.
 */
int32_t param_table_init(void);

/**
 * @brief Define a parameter.
 *
 * @param[in] param_id     Unique parameter ID.
 * @param[in] type         Parameter type.
 * @param[in] default_val  Default value buffer.
 * @param[in] size         Value size in bytes.
 * @return PARAM_OK on success.
 */
int32_t param_define(uint16_t param_id, param_type_t type,
                      const uint8_t *default_val, uint8_t size);

/**
 * @brief Set a parameter value.
 *
 * @param[in] param_id  Parameter ID.
 * @param[in] value     Value buffer.
 * @param[in] size      Value size.
 * @return PARAM_OK on success.
 */
int32_t param_set(uint16_t param_id, const uint8_t *value, uint8_t size);

/**
 * @brief Get a parameter value.
 *
 * @param[in]  param_id  Parameter ID.
 * @param[out] value     Output buffer (at least PARAM_MAX_VALUE_SIZE).
 * @param[out] size      Actual value size.
 * @return PARAM_OK on success.
 */
int32_t param_get(uint16_t param_id, uint8_t *value, uint8_t *size);

/**
 * @brief Save all parameters to NVM.
 * @return PARAM_OK on success.
 */
int32_t param_save_to_nvm(void);

/**
 * @brief Load all parameters from NVM.
 * @return PARAM_OK on success, PARAM_ERR_NVM on CRC failure.
 */
int32_t param_load_from_nvm(void);

/**
 * @brief Reset all parameters to default values.
 * @return PARAM_OK on success.
 */
int32_t param_reset_defaults(void);

/**
 * @brief Get total number of defined parameters.
 * @return Parameter count.
 */
uint32_t param_count(void);

#ifdef __cplusplus
}
#endif

#endif /* PARAMETER_TABLE_H */
