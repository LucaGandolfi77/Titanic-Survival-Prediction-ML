/**
 * @file gr740.ld
 * @brief Linker script for GR740 LEON4FT On-Board Computer.
 *
 * Memory layout:
 *   - Boot PROM:  0x00000000 (256 KB) — Initial boot code
 *   - MRAM:       0x20000000 (8 MB)   — Non-volatile storage
 *   - EEPROM:     0x30000000 (4 MB)   — Firmware backup
 *   - SRAM:       0x40000000 (32 MB)  — Main execution memory
 *
 * After boot, code is loaded from MRAM/PROM into SRAM for execution.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

OUTPUT_FORMAT("elf32-sparc", "elf32-sparc", "elf32-sparc")
OUTPUT_ARCH(sparc)
ENTRY(_start)

/*
 * Memory regions for GR740
 */
MEMORY
{
    prom   (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    mram   (rwx) : ORIGIN = 0x20000000, LENGTH = 8M
    eeprom (rx)  : ORIGIN = 0x30000000, LENGTH = 4M
    sram   (rwx) : ORIGIN = 0x40000000, LENGTH = 32M
}

/*
 * Stack and heap configuration
 * No dynamic heap allocation after init; heap reserved for init only.
 */
_STACK_SIZE = 32K;       /* Main stack size (per core) */
_IRQ_STACK_SIZE = 8K;    /* IRQ stack size (per core)  */
_HEAP_SIZE = 64K;        /* Init-only heap             */
_NUM_CORES = 4;          /* GR740 quad-core            */

SECTIONS
{
    /*
     * .text — Program code in SRAM (loaded from MRAM at boot)
     */
    .text : ALIGN(0x1000)
    {
        _text_start = .;
        *(.text.boot)           /* Boot entry point first */
        *(.text.init)           /* Initialization code    */
        *(.text .text.*)        /* All other code         */
        *(.gnu.linkonce.t.*)
        _text_end = .;
    } > sram AT > mram

    /*
     * .rodata — Read-only data (constants, strings)
     */
    .rodata : ALIGN(8)
    {
        _rodata_start = .;
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
        _rodata_end = .;
    } > sram AT > mram

    /*
     * .data — Initialized data (copied from MRAM to SRAM at startup)
     */
    .data : ALIGN(8)
    {
        _data_start = .;
        *(.data .data.*)
        *(.gnu.linkonce.d.*)
        _data_end = .;
    } > sram AT > mram

    /* Load address of .data section (in MRAM) for startup copy */
    _data_load_start = LOADADDR(.data);

    /*
     * .bss — Uninitialized data (zeroed at startup)
     */
    .bss (NOLOAD) : ALIGN(8)
    {
        _bss_start = .;
        *(.bss .bss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
        _bss_end = .;
    } > sram

    /*
     * .noinit — Variables that must NOT be initialized at startup
     * (preserved across software resets)
     */
    .noinit (NOLOAD) : ALIGN(8)
    {
        _noinit_start = .;
        *(.noinit .noinit.*)
        _noinit_end = .;
    } > sram

    /*
     * .heap — Init-time heap (for RTEMS workspace, etc.)
     */
    .heap (NOLOAD) : ALIGN(8)
    {
        _heap_start = .;
        . += _HEAP_SIZE;
        _heap_end = .;
    } > sram

    /*
     * Stacks — One per core, plus IRQ stacks
     * Stacks grow downward on SPARC.
     */
    .stack (NOLOAD) : ALIGN(16)
    {
        /* Core 0 IRQ stack */
        _irq_stack_start_0 = .;
        . += _IRQ_STACK_SIZE;
        _irq_stack_end_0 = .;

        /* Core 0 main stack */
        _stack_start_0 = .;
        . += _STACK_SIZE;
        _stack_end_0 = .;

        /* Core 1 IRQ stack */
        _irq_stack_start_1 = .;
        . += _IRQ_STACK_SIZE;
        _irq_stack_end_1 = .;

        /* Core 1 main stack */
        _stack_start_1 = .;
        . += _STACK_SIZE;
        _stack_end_1 = .;

        /* Core 2 IRQ stack */
        _irq_stack_start_2 = .;
        . += _IRQ_STACK_SIZE;
        _irq_stack_end_2 = .;

        /* Core 2 main stack */
        _stack_start_2 = .;
        . += _STACK_SIZE;
        _stack_end_2 = .;

        /* Core 3 IRQ stack */
        _irq_stack_start_3 = .;
        . += _IRQ_STACK_SIZE;
        _irq_stack_end_3 = .;

        /* Core 3 main stack */
        _stack_start_3 = .;
        . += _STACK_SIZE;
        _stack_end_3 = .;
    } > sram

    /*
     * NVM Parameter Storage — fixed region in MRAM
     */
    .nvm_params (NOLOAD) : ALIGN(4096)
    {
        _nvm_params_start = .;
        . += 8K;
        _nvm_params_end = .;
    } > mram

    /*
     * Flight Data Recorder — remaining MRAM space
     */
    .flight_log (NOLOAD) : ALIGN(4096)
    {
        _flight_log_start = .;
        . = ORIGIN(mram) + LENGTH(mram);
        _flight_log_end = .;
    } > mram

    /*
     * End of used SRAM marker
     */
    _end = .;
    _sram_end = ORIGIN(sram) + LENGTH(sram);

    /*
     * Discard debugging sections for flight builds
     */
    /DISCARD/ :
    {
        *(.comment)
        *(.note.*)
    }
}

/*
 * Provide symbols for startup code
 */
PROVIDE(_stack_top = _stack_end_0);          /* Default stack pointer (core 0) */
PROVIDE(_trap_table = _text_start);          /* Trap table at start of .text   */
