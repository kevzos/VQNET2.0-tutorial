这个项目是根据source/rst中的api文档：rst文档中介绍的接口，生成mcp。

实现方式 参考rst_to_mcp.py 以及test_simple_server.py实现，输出是一段该接口代码。

测试环境在conda环境 mini311下。

每进行10轮对话，总结当前情况到一个md中以时间：精确到秒的作为文件名。并删除中间temp文件，git commit 相关必要文件，不要提交markdown文件。msg以[feat] [ref] [fix]开始,使用英文。

对rst中理解不清晰的接口，保存到TODO.md里面。

测试代码单独新建一个tests目录下

