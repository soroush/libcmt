AC_INIT([libcmt], [0.1.0], [soroush@phemto-tech.com], [libcmt],
        [https://github.com/soroush/libcmt])
AC_PREREQ([2.59])
AM_INIT_AUTOMAKE([1.10 -Wall no-define])
AM_SILENT_RULES
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AC_LANG([C++])
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11()
AM_PROG_AR
AC_TYPE_SIZE_T
AC_HEADER_STDBOOL

LT_INIT([shared static pic-only])

PKG_CHECK_MODULES([OPENCV], [opencv >= 2.3])

AC_SUBST([CMT_SO_VERSION], [0:0:0])
AC_SUBST([CMT_API_VERSION], [0.1.0])

AC_CONFIG_FILES([Makefile src/Makefile
                 libcmt.pc:libcmt.pc.in])
AC_OUTPUT