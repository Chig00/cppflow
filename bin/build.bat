call bin\clean
mkdir build
g++ src/main/cpp/main.cpp -o build/main -std=c++23 -Ilib^
 -ltensorflow -lmingw32 -lsdl2main -lsdl2 -lsdl2_net || goto :error
goto :success

:error
call bin\clean
exit /b 1

:success
