# modelpool

基于 Python 标准库 `asyncio` 的本地模型句柄生命周期组件。只负责**加载、
停用、借用**三件事；不做推理、持久化或进程崩溃恢复。

## 接口

```python
from modelpool import (
    ModelPool, CapacityExceeded, Superseded, NotActive,
)

pool = ModelPool(
    paths={"m1": "/models/m1.bin"},   # 固定的 model_id -> 本地文件路径
    load=my_load,                     # async load(path) -> handle，可能挂起
    close=my_close,                   # async close(handle)，保证成功，可能挂起
    capacity=5,                       # 全局资源位上限
)

await pool.activate("m1")             # 等待模型可用，不返回句柄
async with pool.lease("m1") as h:     # 借用当前已激活句柄
    h.read(...)
pool.deactivate("m1")                 # 立即撤销激活意图（同步、不等待）
```

### 语义要点

- `activate(model_id)`：已激活直接返回；加载中则并发调用共享同一次加载；
  没有空闲资源位时立即抛 `CapacityExceeded`，不排队。取消某个等待者只结束
  该等待，共享加载与激活意图不受影响。`load` 失败向仍在等待的调用者传播，
  释放资源位，之后可重新激活。
- `deactivate(model_id)`：同步返回，不等待加载/租约/关闭。该代未完成的
  `activate` 以 `Superseded` 结束；重复调用无副作用。停用后再次 `activate`
  会启动新一代加载；迟到的旧句柄只会被清理，不会重新发布。
- `lease(model_id)`：异步上下文管理器，仅当模型当前已激活时借出句柄，
  否则立即抛 `NotActive`。退出（含异常与取消）必定归还租约。同一模型
  允许多个并发租约。

### 保证

- 已取得的租约在退出前始终可用原句柄；句柄在仍有租约时不会被关闭。
- 每个成功加载的句柄，在所属激活被撤销且最后一个租约归还后，最终
  恰好关闭一次。
- 资源位从一次 `load` 开始前占用，到加载失败或对应 `close` 完成才归还；
  停用后的在途加载、未归还租约、挂起中的关闭都计入占用。
- 不同模型的加载/借用/关闭互不串行化；不在 `load`/`close`/租约退出
  期间持有任何全局锁。

## 运行测试

```bash
python -m unittest discover
```

测试后端用临时文件构造真实可读取的句柄，记录加载、关闭与资源占用，
并通过 `asyncio.Event` 门控稳定控制关键操作的交错（不依赖 sleep 碰运气）。
