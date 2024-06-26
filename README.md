# Term Fluid Simulation
![term-fluid.png](https://krseoul.imgtbl.com/i/2024/06/26/667bd7a3f18c0.png)

# 介绍
一个终端流体模拟小玩具，灵感来自[这里](http://www.ioccc.org/2012/endoh1/hint.html)

## 玩法
- 编写一个文本
  - `1234567890-=+*@█`被识别为水
  - `#`被识别为墙
- `./fluid < 你的文本.txt`
- Enjoy yourself!

# 编译
可能需要c++17编译器
```
g++ main.cpp -std=c++20 -O3 -march=native -o fluid
```

# 例子
```
./fluid < hourglass.txt
```
![hourglass.gif](https://vip.helloimg.com/i/2024/06/26/667be13b84b8e.gif)
