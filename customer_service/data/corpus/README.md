# 语料放置说明

受版权保护，小说全文不随仓库分发。运行前请自行准备语料：

1. 将《遥远的救世主》全文保存为 **UTF-8 编码** 的纯文本文件；
2. 放到本目录，文件名固定为：

   ```
   data/corpus/遥远的救世主.txt
   ```

3. 首次启动服务时会自动构建索引（生成 `.index/`，已被 git 忽略）；
4. 更换语料后，加 `--rebuild` 参数强制重建：

   ```bash
   python run_service.py --rebuild --reload
   python run.py --rebuild -q "本书简介"
   ```

> 本目录下的 `*.txt` 已在 `.gitignore` 中忽略，不会被提交。
