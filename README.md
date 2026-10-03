# task26100441

## model_pool：本地模型句柄的异步生命周期组件

纯 `asyncio` 标准库实现，只负责加载、停用与借用，不做推理、持久化或崩溃恢复。

### 用法

```python
import asyncio
from model_pool import ModelPool, CapacityExceeded, Superseded, NotActive

async def load(path: str):
    ...  # 可能长时间挂起；失败时不得遗留资源
    return handle

async def close(handle) -> None:
    ...  # 保证成功；也可能长时间挂起

pool = ModelPool({"llm": "/models/llm.bin"}, load, close, capacity=5)

await pool.activate("llm")            # 等待可用；并发调用共享同一次加载
async with pool.lease("llm") as h:    # 仅借用当前已激活句柄，否则立即 NotActive
    h.read()                          # 退出上下文（含异常/取消）自动归还
await pool.deactivate("llm")          # 立即撤销激活意图；挂起的 activate 以 Superseded 结束
await pool.wait_idle()                # 等待所有在途 load/close 收尾（测试/关停辅助）
```

### 语义要点

- `activate`：已激活直接返回；加载中则共享等待；无空闲资源位立即抛 `CapacityExceeded`，不排队。取消某个等待者只结束该等待，共享加载与激活意图不受影响。
- `deactivate`：立即生效、幂等；不等待加载、租约或关闭完成。旧加载随后成功也只清理，绝不重新发布或覆盖新一代句柄。
- `lease`：同一代可同时多个租约；句柄在仍有租约时绝不会被关闭。撤销激活且最后一个租约归还后，每个成功加载的句柄最终恰好 `close` 一次。
- 容量：从开始 `load` 前占位，到加载失败或对应 `close` 完成才归还；停用后的在途加载、未归还租约、挂起中的关闭都计入。不同模型的加载/借用/关闭互不相串行化。

### 测试

```sh
python -m unittest discover
```

测试后端用临时文件构造真实可读句柄，记录加载/关闭/资源占用，并用 `asyncio.Event` 门控确定性控制关键交错（无 sleep 碰运气）。
