/**
 * @file modes.h
 * @brief Operating mode definitions.
 *
 * BOOT → SAFE → NOMINAL → SCIENCE
 *              ↕ ECLIPSE
 *              ↕ DETUMBLING
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef MODES_H
#define MODES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MODE_BOOT       = 0,   /**< Initial boot, HW init              */
    MODE_SAFE       = 1,   /**< Minimum functionality, safe config  */
    MODE_NOMINAL    = 2,   /**< Normal operations                   */
    MODE_SCIENCE    = 3,   /**< Payload active, data acquisition    */
    MODE_ECLIPSE    = 4,   /**< Eclipse power conservation          */
    MODE_DETUMBLING = 5,   /**< ADCS detumbling after deploy/fault  */
    MODE_COUNT      = 6
} obc_mode_t;

/**
 * @brief Get mode name string.
 */
static inline const char *mode_to_string(obc_mode_t mode)
{
    static const char *names[MODE_COUNT] = {
        "BOOT", "SAFE", "NOMINAL", "SCIENCE", "ECLIPSE", "DETUMBLING"
    };
    if ((uint32_t)mode < (uint32_t)MODE_COUNT) {
        return names[mode];
    }
    return "UNKNOWN";
}

#ifdef __cplusplus
}
#endif

#endif /* MODES_H */
