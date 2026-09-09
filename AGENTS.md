# Working Agreement for Agents

任何修改本仓库的 agent、reviewer 或维护者，先阅读 [`PROJECT_REFERENCE.md`](PROJECT_REFERENCE.md)。它是路线、证据等级、依赖边界和公开声明的单一维护参考。

## 修改前

- 确认修改属于 L4 主线、前沿分支，还是维护工具；不要为了增加文件数量而新增内容。
- 检查受影响的 Notebook、`notebooks/README.md`、`README.md`、`index.html` 和 requirements 是否需要同步。
- 把合成实验、公开 benchmark 结果和系统证据分开，不把 toy result 写成真实车辆能力。

## 修改后

- 运行相关 Notebook；路线或依赖变化时运行全部 20 个 Notebook 的验证。
- 运行 `git diff --check`，检查 HTML 的 Notebook 链接和依赖安装说明。
- 让 `reviewer` 与 `devil's advocate` 独立审查；实现者不得替代任一角色签字。
- 更新 `PROJECT_REFERENCE.md` 的缺口和变更日志，并在 [review/](review/) 中留下审查结果。

## 角色入口

- [Reviewer](review/roles/reviewer.md)：正确性、完整性、可运行性和岗位相关性。
- [Devil's Advocate](review/roles/devils-advocate.md)：反方质疑、过度承诺、错误能力感和招聘经理视角。
- [Review protocol](review/README.md)：触发条件、输出格式、优先级和合并门禁。

