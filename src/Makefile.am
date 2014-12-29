ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4
lib_LTLIBRARIES = libcmt.la
libcmt_la_SOURCES = \
	cmt.cpp
libcmt_la_LIBADD = $(OPENCV_LIBS)
libcmt_la_LDFLAGS = -version-info $(CMT_SO_VERSION)
cmt_includedir = $(includedir)/cmt
cmt_include_HEADERS = \
	cmt.hpp
