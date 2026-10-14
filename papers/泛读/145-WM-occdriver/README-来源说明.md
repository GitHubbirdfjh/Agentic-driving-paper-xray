# OccDriver PDF 获取说明

- OpenReview 直链 https://openreview.net/pdf/903be19d414df075445f33c7634a785f5ac24c9a.pdf
  直接下载返回 HTTP 403（2026-10-11 实测，curl 与浏览器 UA 均失败）。
- 精读依据：经由 OpenReview 全文文本提取（1301 行，含摘要/引言/方法/公式/实验表/消融/结论/参考文献），
  覆盖正文全部章节；附录 A.2/C/D/E/F 未能获取，文档中已标注哪些细节因此无法核实。
- venue 核实：OpenReview 页面明确标注 "Published as a conference paper at ICLR 2026"。
- 官方代码：全网检索未发现作者发布的官方仓库，文档中如实标注"未找到官方代码"。
