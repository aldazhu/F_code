new low strategy
```txt
highest_window:=15;
lowest_window:=50;
ema_period:=5;
ema_sell_period:=10;


high:=HHV(HIGH, highest_window);
low:=LLV(LOW, lowest_window);
ema:=EMA(CLOSE, ema_period);
ema_sell:=EMA(CLOSE, ema_sell_period);


buy:=CROSS(CLOSE, ema) AND REF(LOW, 1) > low AND CLOSE > OPEN;


sell:=CROSS(high, REF(CLOSE, 1));

STICKLINE(buy, LOW, HIGH, 0.7, 1), COLORRED;
STICKLINE(sell, LOW, HIGH, 0.7, 1), COLORGREEN;
```