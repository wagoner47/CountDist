set(UNIT_TEST_LIST calc_distances)

foreach(NAME IN LISTS UNIT_TEST_LIST)
	message(STATUS "Test file for: ${NAME}")
	list(APPEND UNIT_TEST_SOURCE_LIST "test_${NAME}_cpp.cpp")
	list(APPEND UNIT_TEST_HEADER_LIST "${NAME}.h")
endforeach()

set(TARGET_NAME tests)

add_executable(${TARGET_NAME} "test_cpp_main.cpp" ${UNIT_TEST_SOURCE_LIST})

target_include_directories(${TARGET_NAME} PUBLIC external/catch2/ ${CountDistCPP_SOURCE_DIR}/include/v${vers})

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
