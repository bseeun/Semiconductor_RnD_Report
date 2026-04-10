# 내부 RAG 코퍼스 (파일명 정본)

적재 시 파일명 규칙: `{company}_{technology}_{source_type}_{YYYYMM}.pdf`  
(`document_metadata.py`의 `VALID_*` 목록과 일치해야 함.)

| 원본 파일명 | 변경 파일명 |
| --- | --- |
| JESD270-4A.pdf | `JEDEC_HBM4_standard_202401.pdf` |
| HC35_SKhynix_DSM_YongkeeKwon_rev3.pdf | `SKHynix_PIM_whitepaper_202308.pdf` |
| HBM2-PIM(Processing-in-Memory)_Samsung.pdf | `Samsung_PIM_whitepaper_202102.pdf` |
| cxl-memory-expansion-a-close-look-on-actual-platform_Micron.pdf | `Micron_CXL_whitepaper_202309.pdf` |
| Samsung_CMM-D_Utilization_in_IMDB_Applications.pdf | `Samsung_CXL_whitepaper_202311.pdf` |
| Dynamic_Capacity_Service_for_Improving_CXL_Pooled_Memory_Efficiency.pdf | `SKHynix_CXL_whitepaper_202401.pdf` |
| 삼성전자 사업보고서(일반법인) 등 | `Samsung_ALL_ir_report_202503.pdf` |

적재:

```bash
python ingest.py --dir docs/pdfs
```

코퍼스·메타 규칙은 **이 파일만** 정본으로 두고, 다른 문서에서는 링크로 참조합니다.
