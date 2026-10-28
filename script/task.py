from clearml import Task

task = Task.init(
    project_name="Алгоритмы ИИ и ML в дизайне противоопухолевых лекарств",
    task_name="Connection Test",
    reuse_last_task_id=False
)

print(task.get_output_log_web_page())
task.close()