# ============================================================================
#  GR740 LEON4FT OBC Flight Software — Makefile
# ============================================================================
#
#  Cross-compilation for SPARC/LEON4 using BCC2 (sparc-rtems5-gcc).
#
#  Targets:
#    all       — Build the ELF binary
#    clean     — Remove build artefacts
#    flash     — Upload via GRMON (requires grmon connected)
#    size      — Print section sizes
#    dump      — Objdump disassembly
#    crc       — Compute CRC-32 of output binary
#    test      — (Host) Build & run unit tests with native GCC
#
#  Usage:
#    make                       # build flight image
#    make CROSS=sparc-rtems5-   # explicit cross prefix
#    make clean
#    make flash
#    make test                  # host unit tests (native gcc)
#
# ============================================================================

# ── Toolchain ──────────────────────────────────────────────────────────────
CROSS       ?= sparc-rtems5-
CC           = $(CROSS)gcc
LD           = $(CROSS)gcc
AS           = $(CROSS)as
OBJCOPY      = $(CROSS)objcopy
OBJDUMP      = $(CROSS)objdump
SIZE         = $(CROSS)size
AR           = $(CROSS)ar
NM           = $(CROSS)nm

# Host compiler for unit tests
HOST_CC      = gcc

# ── Directories ────────────────────────────────────────────────────────────
SRCDIR       = .
BUILDDIR     = build
OBJDIR       = $(BUILDDIR)/obj
DEPDIR       = $(BUILDDIR)/dep
TESTBUILDDIR = $(BUILDDIR)/tests

# ── Output ─────────────────────────────────────────────────────────────────
TARGET       = gr740_obc_fsw
ELF          = $(BUILDDIR)/$(TARGET).elf
BIN          = $(BUILDDIR)/$(TARGET).bin
MAP          = $(BUILDDIR)/$(TARGET).map
LST          = $(BUILDDIR)/$(TARGET).lst

# ── Linker script ──────────────────────────────────────────────────────────
LDSCRIPT     = linker/gr740.ld

# ── Compiler flags ─────────────────────────────────────────────────────────
CFLAGS       = -mcpu=leon3
CFLAGS      += -msoft-float
CFLAGS      += -std=c99
CFLAGS      += -O2
CFLAGS      += -Wall -Wextra -Wpedantic
CFLAGS      += -Wstrict-prototypes -Wmissing-prototypes
CFLAGS      += -Wcast-align -Wpointer-arith
CFLAGS      += -Wshadow -Wundef
CFLAGS      += -Wno-unused-parameter
CFLAGS      += -fno-common
CFLAGS      += -ffunction-sections -fdata-sections
CFLAGS      += -ffreestanding

# Include paths
CFLAGS      += -I$(SRCDIR)
CFLAGS      += -I$(SRCDIR)/config
CFLAGS      += -I$(SRCDIR)/bsp
CFLAGS      += -I$(SRCDIR)/drivers
CFLAGS      += -I$(SRCDIR)/middleware
CFLAGS      += -I$(SRCDIR)/fsw

# ── Assembler flags ───────────────────────────────────────────────────────
ASFLAGS      = -mcpu=leon3

# ── Linker flags ───────────────────────────────────────────────────────────
LDFLAGS      = -mcpu=leon3
LDFLAGS     += -msoft-float
LDFLAGS     += -T $(LDSCRIPT)
LDFLAGS     += -nostartfiles
LDFLAGS     += -Wl,--gc-sections
LDFLAGS     += -Wl,-Map=$(MAP)
LDFLAGS     += -Wl,--cref

# ── Source files ───────────────────────────────────────────────────────────

# BSP
SRC_BSP      = bsp/gr740_init.c \
               bsp/irq_handler.c

# Drivers
SRC_DRV      = drivers/can/grcan.c \
               drivers/spacewire/grspw.c \
               drivers/spacewire/rmap.c \
               drivers/uart/apbuart.c \
               drivers/spi/spictrl.c \
               drivers/i2c/i2cmst.c \
               drivers/gpio/grgpio.c \
               drivers/timer/gptimer.c

# Middleware — CCSDS
SRC_CCSDS    = middleware/ccsds/space_packet.c \
               middleware/ccsds/tm_framing.c \
               middleware/ccsds/tc_framing.c

# Middleware — PUS
SRC_PUS      = middleware/pus/pus_st01.c \
               middleware/pus/pus_st03.c \
               middleware/pus/pus_st05.c \
               middleware/pus/pus_st08.c \
               middleware/pus/pus_st09.c \
               middleware/pus/pus_st11.c \
               middleware/pus/pus_st17.c

# Middleware — Routing
SRC_ROUTE    = middleware/routing/packet_router.c

# FSW — Core
SRC_FSW      = fsw/scheduler/minor_frame.c \
               fsw/scheduler/task_table.c \
               fsw/watchdog/watchdog.c \
               fsw/watchdog/health_monitor.c \
               fsw/fdir/fdir.c \
               fsw/modes/mode_manager.c \
               fsw/obc_tasks.c \
               fsw/obc_main.c

# FSW — Support
SRC_SUPPORT  = fsw/storage/nvm_manager.c \
               fsw/storage/parameter_table.c \
               fsw/thermal/thermal_control.c \
               fsw/power/eps_interface.c \
               fsw/adcs/adcs_interface.c

# All sources
SRCS         = $(SRC_BSP) $(SRC_DRV) $(SRC_CCSDS) $(SRC_PUS) \
               $(SRC_ROUTE) $(SRC_FSW) $(SRC_SUPPORT)

# Object files
OBJS         = $(patsubst %.c,$(OBJDIR)/%.o,$(SRCS))

# Dependency files
DEPS         = $(patsubst %.c,$(DEPDIR)/%.d,$(SRCS))

# ── Test sources ───────────────────────────────────────────────────────────
TEST_SRCS    = tests/test_ccsds.c \
               tests/test_watchdog.c \
               tests/test_fdir.c \
               tests/test_pus.c \
               tests/test_loop.c

TEST_BINS    = $(patsubst tests/%.c,$(TESTBUILDDIR)/%,$(TEST_SRCS))

# Host test compiler flags
HOST_CFLAGS  = -std=c99 -O0 -g -Wall -Wextra -I$(SRCDIR)

# ============================================================================
#  RULES
# ============================================================================

.PHONY: all clean flash size dump crc test info

all: $(ELF) $(BIN) size

# ── Link ───────────────────────────────────────────────────────────────────
$(ELF): $(OBJS) $(LDSCRIPT)
	@echo "  LD      $@"
	@mkdir -p $(dir $@)
	$(LD) $(LDFLAGS) $(OBJS) -o $@

# ── Binary image ──────────────────────────────────────────────────────────
$(BIN): $(ELF)
	@echo "  OBJCOPY $@"
	$(OBJCOPY) -O binary $< $@

# ── Compile C ──────────────────────────────────────────────────────────────
$(OBJDIR)/%.o: %.c
	@echo "  CC      $<"
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)/$*.d)
	$(CC) $(CFLAGS) -MMD -MF $(DEPDIR)/$*.d -c $< -o $@

# ── Size report ────────────────────────────────────────────────────────────
size: $(ELF)
	@echo ""
	@echo "  === Section Sizes ==="
	$(SIZE) -A -d $<
	@echo ""

# ── Disassembly ────────────────────────────────────────────────────────────
dump: $(ELF)
	$(OBJDUMP) -d -S $< > $(LST)
	@echo "  Disassembly written to $(LST)"

# ── CRC-32 of binary ──────────────────────────────────────────────────────
crc: $(BIN)
	@echo "  CRC-32:"
	@cksum $< || crc32 $< || echo "  (no cksum/crc32 tool found)"

# ── Flash via GRMON ────────────────────────────────────────────────────────
flash: $(ELF)
	@echo "  FLASH   $< via GRMON"
	grmon -uart /dev/ttyUSB0 -baud 115200 \
	      -c "load $<" \
	      -c "verify $<" \
	      -c "run" \
	      -c "exit"

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	@echo "  CLEAN"
	rm -rf $(BUILDDIR)

# ── Info ───────────────────────────────────────────────────────────────────
info:
	@echo "  CROSS   = $(CROSS)"
	@echo "  CC      = $(CC)"
	@echo "  CFLAGS  = $(CFLAGS)"
	@echo "  LDFLAGS = $(LDFLAGS)"
	@echo "  SRCS    = $(SRCS)"
	@echo "  OBJS    = $(OBJS)"

# ============================================================================
#  HOST UNIT TESTS (native gcc — no RTEMS / HW deps)
# ============================================================================

test: $(TEST_BINS)
	@echo ""
	@echo "  === Running Unit Tests ==="
	@pass=0; fail=0; \
	for t in $(TEST_BINS); do \
	    echo "  RUN     $$t"; \
	    if $$t; then \
	        pass=$$((pass+1)); \
	    else \
	        fail=$$((fail+1)); \
	        echo "  FAIL    $$t"; \
	    fi; \
	done; \
	echo ""; \
	echo "  Tests passed: $$pass"; \
	echo "  Tests failed: $$fail"; \
	if [ $$fail -ne 0 ]; then exit 1; fi

# Each test binary links against the relevant source files.
# Since tests stub out BSP/driver code, we compile only the
# middleware and FSW modules needed.

$(TESTBUILDDIR)/test_ccsds: tests/test_ccsds.c \
	middleware/ccsds/space_packet.c \
	middleware/ccsds/tm_framing.c \
	middleware/ccsds/tc_framing.c
	@echo "  HOST_CC $@"
	@mkdir -p $(TESTBUILDDIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $^

$(TESTBUILDDIR)/test_watchdog: tests/test_watchdog.c \
	fsw/watchdog/watchdog.c \
	fsw/watchdog/health_monitor.c
	@echo "  HOST_CC $@"
	@mkdir -p $(TESTBUILDDIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $^

$(TESTBUILDDIR)/test_fdir: tests/test_fdir.c \
	fsw/fdir/fdir.c
	@echo "  HOST_CC $@"
	@mkdir -p $(TESTBUILDDIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $^

$(TESTBUILDDIR)/test_pus: tests/test_pus.c \
	middleware/pus/pus_st01.c \
	middleware/pus/pus_st03.c \
	middleware/pus/pus_st05.c \
	middleware/pus/pus_st08.c \
	middleware/pus/pus_st09.c \
	middleware/pus/pus_st11.c \
	middleware/pus/pus_st17.c \
	middleware/ccsds/space_packet.c
	@echo "  HOST_CC $@"
	@mkdir -p $(TESTBUILDDIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $^

$(TESTBUILDDIR)/test_loop: tests/test_loop.c \
	middleware/routing/packet_router.c \
	middleware/pus/pus_st01.c \
	middleware/pus/pus_st17.c \
	middleware/ccsds/space_packet.c
	@echo "  HOST_CC $@"
	@mkdir -p $(TESTBUILDDIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $^

# ── Include auto-generated dependencies ────────────────────────────────────
-include $(DEPS)
