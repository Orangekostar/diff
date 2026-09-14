# 新增图有限QA复核

Fig1/Fig4最终PDF均经nature-figure字体和碰撞检查，最小源图文字8pt，0碰撞失败/0警告。已实际查看两张PNG及其主稿页面。主稿宽166mm，源图约183mm，字体随排版缩小约9%，仍可读。未修改的39复用文件与原件SHA一致，不重跑其全部QA。

源码静态检查原报告保留。EXPORT-VECTOR与EXPORT-RASTER的两个FAIL是检查器未解析save函数的动态扩展名循环；实际PDF/SVG/PNG六文件存在，PDF由编译及渲染验证，不把原报告改成PASS。300dpi警告对照本轮FIGURE_CONTRACT.md的300dpi栅格预览目标接受；论文采用矢量PDF，不依赖栅格预览分辨率。

流程图只连接已请求内部描述子，没有隐藏完整X、真实y或语言规划进入Actor的连线。时机图保留全部36个有符号数值，无新误差条。修改处视觉复核同时发现Fig5图注曾描述未显示的“arrows and predictions”，已改为实际显示的下一格轮廓及观测状态；复用图像本身未动。
