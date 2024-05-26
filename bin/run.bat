build\main %1 %2 || goto :error
goto :success

:error
echo.
echo ERROR - APPLICATION TERMINATION

:success
