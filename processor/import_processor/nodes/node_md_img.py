import base64
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from sys import prefix
from typing import Tuple, List, Dict, Deque

import minio
from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject
from openai import OpenAI
from config.lm_config import lm_config
from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行···")

        #1.参数处理
        md_content ,md_path_obj,images_dir = self._step_1_get_content(state)

        #2.图片扫描
        target_images = self._step_2_scan_images(md_content,images_dir)

        #3.视觉模型摘要
        summaries = self._step_3_qenerate_summarize(md_path_obj , target_images)

        #4.上传minio，替代md
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem , target_images, summaries ,md_content)

        #5.保存备份md
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'] , new_md_content)

        # 6.更新state状态值
        state["md_content"] =new_md_content
        state["md_path"] = new_md_file_name
        return state

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        从全局状态中提取并初始化MD处理所需核心数据
        :param state: 流程全局状态对象
        :return: 元组(MD文件内容, MD文件路径, 图片文件夹路径)
        :raise FileProcessingError: 当状态中无有效MD文件路径时抛出
        """

        # 1、参数非空校验
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name='md_path', expected_type=str)

        # 2、路径转换
        md_path_obj = Path(md_path)

        # 3、检查PDF文件的有效性
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"MD文件{md_path_obj.name}不存在")

        # 4、获取md_content
        md_content = state["md_content"]

        # 5、组装图片文件夹路径：图片文件夹固定为MD文件同级的images目录
        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
        :param md_content: MD文件完整内容
        :param images_dir: 图片文件夹路径对象
        :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
        """
        # 1. 定义待处理图片列表
        target_images = []

        # 2. 遍历图片文件夹
        for image_file in os.listdir(images_dir):

            # 2.1 过滤无效后缀
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue

            # 1.2 组装图片完整路径并转成字符串
            img_path = str(images_dir / image_file)

            # 1.3 查找图片在MD中的引用上下文
            context = self._find_image_in_md(md_content, image_file)

            # 过滤MD中未引用的图片
            if not context:
                self.logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
                continue

            # 1.4 组装待处理图片元数据，取第一个匹配的图片上下文
            target_images.append((image_file, img_path, context))

        return target_images

    #步骤2的上下文查找：在md文件中查找图片位置
    def _find_image_in_md(self, md_content: str, image_file: str, context_len: int = 100) -> Tuple[str, str]:
        """
        查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
        :param md_content: MD文件完整内容
        :param image_file: 图片文件名（含后缀）
        :param context_len: 上下文截取长度，默认前后各100字符
        :return: 每个图片的(上文, 下文)元组，无匹配则返回None
        """

        # 1、定义正则表达式
        # ![描述](images/文件名.扩展名)
        # r"字符串"：不要将其中的特殊符号进行转义
        # re.escape 转义图片文件名中的特殊字符，避免正则语法错误
        # .* 贪婪匹配 .*? 非贪婪匹配
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

        # 2、找到1个匹配项即返回
        match = pattern.search(md_content)
        if not match:
            return None  # 没有找到

        # 3、截取匹配位置的上文和下文（防止索引越界）
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]

        # 4、返回上下文元组
        return pre_text, post_text

    def _step_3_qenerate_summarize(self, doc_stem :str , target_images : List[Tuple[str,str,Tuple[str,str]]])->Dict[str,str]:
        """
         步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
         :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
         :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
         :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
         :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
         """
        summaries = {}

        # 1、外部初始化双端队列，用于API速率限制，跨循环复用
        request_deque = deque()

        # 2、循环处理图片
        for img_file, image_path, context in target_images:
            # 2.1、速率限制（时间窗口限流)
            self._apply_api_rate_limit(request_deque, max_requests=10)

            # 2.2、调用大模型生成图片摘要
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)

        return summaries

    #步骤3的速率限制方法
    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")

    #步骤3调用大模型方法
    def _summarize_image(self, image_path, root_folder, image_content: List[Tuple[str,str,Tuple[str,str]]])->str:
        #1. llm 模型工具
        # ai = ChatOpenAI(
            #     model=lm_config.vl_model,
            #     temperature=lm_config.temperature,
            #     base_url = lm_config.base_url,
            #     api_key = lm_config.api_key,
            #     model_kwargs= {"response_format":{"type":"json_object"}}
            # )
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        try:
            vl_ai = get_llm_client(lm_config.vl_model)

            #2.调用模型
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。尽量不要超过10个字"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            response = vl_ai.invoke(messages)
            return response.content.strip().replace("\n", "")

        except Exception as e:
            self.logger.error(f"图像总结失败：{image_path}, 错误{e}")
            return "图片描述"

    #步骤4：将图片上传到目标文件夹，并替换Markdown文档中的图片链接
    def _step_4_upload_and_replace(self, doc_stem :str, target_images : List[Tuple[str,str,Tuple[str,str]]], summaries : Dict[str ,str], md_content :str )->str:
        """
        步骤4：将图片上传到目标文件夹，并替换Markdown文档中的图片链接
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param summaries: 图片摘要字典，键：图片文件名，值：图片内容摘要
        :param md_content: Markdown文档内容
        :return: 替换后的Markdown文档内容
        """
        #0. minio客户端
        minio_client = get_minio_client()
        # 构造上传目录，去除文件名中的空格
        minio_img_dir = minio_config.img_dir
        upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

        #1.清理minio目录
        self._clean_minio_directory(minio_client, upload_dir)

        #2.批量上传图片，获取minio的urls
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)

        #3.将摘要和url路径合并
        image_info = self.merge_summary_and_url(summaries,urls)

        #4.替换md中的摘要和路径
        md_content = self._process_summart_and_url(md_content, image_info)

    # 步骤4方法1：清理minio目录
    def _clean_minio_directory(self, minio_client :Minio, upload_dir)-> None:

        """
        幂等性清理：上传前先删除 MinIO 中指定目录下的旧文件。
        防止重名文件导致的内容混淆或垃圾堆积。
        :param minio_client: 初始化完成的MinIO客户端对象
        :param prefix: MinIO目录前缀（要清理的目录路径）
        """
        try:
            objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=prefix, recursive=True)
            # 构造删除列表
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
                for error in errors:
                    self.logger.error(f"删除失败：{error}")
        except Exception as e:
            self.logger.error(f"清理minio目录失败：{e}")

    # 步骤4方法2：批量上传图片，获取minio的urls
    def _upload_images_batch(self, minio_client : minio , upload_dir :str, target_images :list[Tuple[str,str,Tuple[str,str]]]):
        urls = {}


    # 步骤4方法3：将摘要和url路径合并
    def merge_summary_and_url(self, summaries, urls):
        pass

    # 步骤4方法4：替换md中的摘要和路径
    def _process_summart_and_url(self, md_content, image_info):
        pass


if __name__ == '__main__':
    setup_logging()

    init_state={
        "md_path": "F:\output\hak180产品安全手册\hak180产品安全手册.md",
        "md_content" : None
    }

    node = NodeMDImg()
    result = node(init_state)

    #打印结果
    dump = json.dumps(result)
    print(dump)

