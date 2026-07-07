# General Capability Layer 通用能力层

通用能力层提供所有业务 Skill 都可以复用的能力。

## 当前能力槽位

- Web Search；
- Web Page Reader；
- Weather；
- Time / Date；
- Calculator；
- Unit Converter；
- File Parser；
- Knowledge Search；
- News Search；
- Translation；
- Summarizer。

## 当前实现状态

部分能力是本地可运行实现，例如：

- `calculator.py`；
- `time_date.py`；
- `summarizer.py`；
- `unit_converter.py`。

需要外部 provider 的能力当前是 adapter/mock，例如：

- `web_search.py`；
- `news_search.py`；
- `weather.py`；
- `file_parser.py`。

生产接入时只需要替换 adapter 内部实现，不需要改 Agent Core 的路由边界。

