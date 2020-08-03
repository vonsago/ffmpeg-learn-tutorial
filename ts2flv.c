//
// Created by von on 2020/7/31.
//
#include <stdio.h>
#include <string.h>

#define PRINTFPAT_INFO 1

#define PAT_PID 0x0000
#define PAT_TABLE_ID 0x00
#define DEFAULT_TABLE_ID 0x80

#define DIT_PID 0x001E
#define DIT_TABLE_ID 0x7E
#define INITIAL_VERSION 0xff
#define SECTION_COUNT_256 256
#define SECTION_MAX_LENGTH_4096 4096

#define PMT_VUDIO_COUNT 32
#define PMT_DESCRIPTOR_MAX 256
#define PMT_PROGRAM_DESCRIPTOR_MAX 1024
#define PAT_PROGARM_MAX 512

#define PROGRAM_MAX 128
#define CA_SYSTEM_MAX 64

typedef struct PMT_CAT_INFO_T {
    unsigned int uiPMT_CA_system_id: 16;
    unsigned int uiPMT_CA_PID: 13;
} PMT_CAT_INFO_T;

typedef struct PMT_INFO_T {
    unsigned int uiProgramNumber: 16;
    unsigned int uiVideoPID: 13;
    unsigned int uiAudioPID[PMT_VUDIO_COUNT];
} PMT_INFO_T;

typedef struct CAT_INFO_T
{
    unsigned int uiCA_system_id :16;
    unsigned int uiCA_PID :13;
} CAT_INFO_T;

typedef struct PAT_INFO_T {
    unsigned int uiProgramNumber: 16;
    unsigned int uiPMT_PID: 13;
} PAT_INFO_T;

typedef struct TS_PMT_STREAM_T {
    unsigned int uiStream_type: 8;
    unsigned int uiReserved_fifth: 3;
    unsigned int uiElementary_PID: 13;
    unsigned int uiReserved_sixth: 4;
    unsigned int uiES_info_length: 12;
    unsigned char aucDescriptor[PMT_DESCRIPTOR_MAX];
} TS_PMT_STREAM_T;

typedef struct TS_PMT_T {
    unsigned int uiTable_id: 8;
    unsigned int uiSection_syntax_indicator: 1;
    unsigned int uiZero: 1;
    unsigned int uiReserved_first: 2;
    unsigned int uiSection_length: 12;
    unsigned int uiProgram_number: 16;
    unsigned int uiReserved_second: 2;
    unsigned int uiVersion_number: 5;
    unsigned int uiCurrent_next_indicator: 1;
    unsigned int uiSection_number: 8;
    unsigned int uiLast_section_number: 8;
    unsigned int uiReserved_third: 3;
    unsigned int uiPCR_PID: 13;
    unsigned int uiReserved_fourth: 4;
    unsigned int uiProgram_info_length: 12;
    unsigned char aucProgramDescriptor[PMT_PROGRAM_DESCRIPTOR_MAX];
    TS_PMT_STREAM_T stPMT_Stream[PMT_DESCRIPTOR_MAX];
    unsigned long uiCRC_32: 32;
} TS_PMT_T;

typedef struct TS_PAT_PROGRAM_T {
    unsigned int uiProgram_number: 16;
    unsigned int uiReserved_third: 3;
    unsigned int uiProgram_map_PID: 13;
} TS_PAT_PROGRAM_T;

typedef struct TS_PAT_T {
    unsigned int uiTable_id: 8;
    unsigned int uiSection_syntax_indicator: 1;
    unsigned int uiZero: 1;
    unsigned int uiReserved_first: 2;
    unsigned int uiSection_length: 12;
    unsigned int uiTransport_stream_id: 16;
    unsigned int uiReserved_second: 2;
    unsigned int uiVersion_number: 5;
    unsigned int uiCurrent_next_indicator: 1;
    unsigned int uiSection_number: 8;
    unsigned int uiLast_section_number: 8;

    TS_PAT_PROGRAM_T stPAT_Program[PAT_PROGARM_MAX];

    unsigned int uiNetwork_PID: 13;
    unsigned long uiCRC_32: 32;
} TS_PAT_T;

typedef struct SECTION_HEAD_T {
    unsigned int uiTable_id: 8;
    unsigned int uiSection_syntax_indicator: 1;
    unsigned int uiZero: 1;
    unsigned int uiReservedFirst: 2;
    unsigned int uiSection_Length: 12;
    unsigned int uiTransport_stream_id: 16;
    unsigned int uiReservedSecond: 2;
    unsigned int uiVersion_number: 5;
    unsigned int uiCurrent_next_indicator: 1;
    unsigned int uiSection_number: 8;
    unsigned int uiLast_section_number: 8;
} SECTION_HEAD_T;

typedef struct TS_PACKAGE_HEAD_T {
    unsigned int uiSync_byte: 8;
    unsigned int uiTransport_error_indicator: 1;
    unsigned int uiPayload_unit_start_indicator: 1;
    unsigned int uiTransport_priority: 1;
    unsigned int uiPID: 13;
    unsigned int uiTransport_scrambling_control: 2;
    unsigned int uiAdapptation_field_control: 2;
    unsigned int uiContinuity_counter: 4;
} TS_PACKAGE_HEAD_T;

int GetTheLoadBeginPostion(unsigned char *pucPackageBuffer);

/******************************************
 *
 * 判断是否已获取
 *
 ******************************************/
int IsSectionGetBefore(unsigned char *pucSectionBuffer, unsigned int *puiRecordSectionNumber) {
    int iLoopTime = 1;
    int iLength = (int) puiRecordSectionNumber[0];
    unsigned int uiSectionNumber = pucSectionBuffer[6];

    for (iLoopTime = 1; iLoopTime < iLength + 1; iLoopTime++) {
        if (puiRecordSectionNumber[iLoopTime] == uiSectionNumber) {
            return 1;
        }
    }
    puiRecordSectionNumber[iLoopTime] = uiSectionNumber;
    puiRecordSectionNumber[0]++;
    return 0;
}

/*********************************************
 * 输出TS_HEAD信息
 *********************************************/
static void PrintTS_PES_HEAD(TS_PACKAGE_HEAD_T stTS_PackageHead) {
    printf("TS_PackageHead.Sync_byte: 0x%02x\n", stTS_PackageHead.uiSync_byte);
    printf("TS_PackageHead.Transport_error_indicator: 0x%02x\n", stTS_PackageHead.uiTransport_error_indicator);
    printf("TS_PackageHead.Payload_unit_start_indicator: 0x%02x\n", stTS_PackageHead.uiPayload_unit_start_indicator);
    printf("TS_PackageHead.Transport_priority: 0x%02x\n", stTS_PackageHead.uiTransport_priority);
    printf("TS_PackageHead.PID: 0x%02x\n", stTS_PackageHead.uiPID);
    printf("TS_PackageHead.Transport_scrambling_control: 0x%02x\n", stTS_PackageHead.uiTransport_scrambling_control);
    printf("TS_PackageHead.Adapptation_field_control: 0x%02x\n", stTS_PackageHead.uiAdapptation_field_control);
    printf("TS_PackageHead.Continuity_counter: 0x%02x\n", stTS_PackageHead.uiContinuity_counter);
}

/******************************************
 *
 * 输出PAT表信息
 *
 ******************************************/
void PrintPAT(TS_PAT_T *pstTS_PAT_T, int iPAT_ProgramCount) {
    printf("\n\n\n");
    printf("-------------PAT info start-------------\n");
    printf("PAT->Table_id: 0x%02x\n", pstTS_PAT_T->uiTable_id);
    printf("PAT->Section_syntax_indicator: 0x%02x\n", pstTS_PAT_T->uiSection_syntax_indicator);
    printf("PAT->Zero: 0x%02x\n", pstTS_PAT_T->uiZero);
    printf("PAT->Reserved_first: 0x%02x\n", pstTS_PAT_T->uiReserved_first);
    printf("PAT->Section_length: 0x%02x\n", pstTS_PAT_T->uiSection_length);
    printf("PAT->Transport_stream_id: 0x%02x\n", pstTS_PAT_T->uiTransport_stream_id);
    printf("PAT->Reserved_second: 0x%02x\n", pstTS_PAT_T->uiReserved_second);
    printf("PAT->Version_number: 0x%02x\n", pstTS_PAT_T->uiVersion_number);
    printf("PAT->Current_next_indicator: 0x%02x\n", pstTS_PAT_T->uiCurrent_next_indicator);
    printf("PAT->Section_number: 0x%02x\n", pstTS_PAT_T->uiSection_number);
    printf("PAT->Last_section_number: 0x%02x\n", pstTS_PAT_T->uiLast_section_number);
    printf("PAT->CRC_32: 0x%08lx\n", pstTS_PAT_T->uiCRC_32);

    int iLoopTime = 0;
    for (iLoopTime = 0; iLoopTime < iPAT_ProgramCount; iLoopTime++) {
        printf("PAT->PAT_Program[%d].Program_number: 0x%02x\n", iLoopTime,
                    pstTS_PAT_T->stPAT_Program[iLoopTime].uiProgram_number);
        if (0 == pstTS_PAT_T->stPAT_Program[iLoopTime].uiProgram_number) {
            printf("PAT->uiNetwork_PID: 0x%02x\n", pstTS_PAT_T->uiNetwork_PID);
        } else {
            printf("PAT->PAT_Program[%d].Reserved_third: 0x%02x\n", iLoopTime,
                        pstTS_PAT_T->stPAT_Program[iLoopTime].uiReserved_third);
            printf("PAT->PAT_Program[%d].Program_map_PID: 0x%02x\n", iLoopTime,
                        pstTS_PAT_T->stPAT_Program[iLoopTime].uiProgram_map_PID);
        }
    }
    printf("-------------PAT info end-------------\n\n");
}

/*********************************************
 * 解析TS的包头
 *********************************************/
static void ParseTS_PackageHead(TS_PACKAGE_HEAD_T *pstTS_PackageHead, unsigned char *pucPackageBuffer) {
    pstTS_PackageHead->uiSync_byte = pucPackageBuffer[0];
    pstTS_PackageHead->uiTransport_error_indicator = pucPackageBuffer[1] >> 7;
    pstTS_PackageHead->uiPayload_unit_start_indicator = (pucPackageBuffer[1] >> 6) & 0x01;
    pstTS_PackageHead->uiTransport_priority = (pucPackageBuffer[1] >> 5) & 0x01;
    pstTS_PackageHead->uiPID = ((pucPackageBuffer[1] & 0x1F) << 8) | pucPackageBuffer[2];
    pstTS_PackageHead->uiTransport_scrambling_control = pucPackageBuffer[3] >> 6;
    pstTS_PackageHead->uiAdapptation_field_control = (pucPackageBuffer[3] >> 4) & 0x03;
    pstTS_PackageHead->uiContinuity_counter = pucPackageBuffer[3] & 0x0f;
}

/*********************************************
 * 解析PAT Section头部部分数据
 *********************************************/
void ParsePATSectionHeader(SECTION_HEAD_T *pstSectionHead, unsigned char *pucPackageBuffer) {
    int iPayloadPosition = -1;

    iPayloadPosition = GetTheLoadBeginPostion(pucPackageBuffer);
    pstSectionHead->uiTable_id = pucPackageBuffer[iPayloadPosition];
    pstSectionHead->uiSection_syntax_indicator = pucPackageBuffer[1 + iPayloadPosition] >> 7;
    pstSectionHead->uiZero = (pucPackageBuffer[1 + iPayloadPosition] >> 6) & 0x1;
    pstSectionHead->uiReservedFirst = (pucPackageBuffer[1 + iPayloadPosition] >> 4) & 0x3;
    pstSectionHead->uiSection_Length =
            (pucPackageBuffer[1 + iPayloadPosition] & 0xf) << 8 | (pucPackageBuffer[2 + iPayloadPosition]);
    pstSectionHead->uiTransport_stream_id =
            pucPackageBuffer[3 + iPayloadPosition] << 8 | pucPackageBuffer[4 + iPayloadPosition];
    pstSectionHead->uiReservedSecond = pucPackageBuffer[5 + iPayloadPosition] >> 6;
    pstSectionHead->uiVersion_number = (pucPackageBuffer[5 + iPayloadPosition] >> 1) & 0x1f;
    pstSectionHead->uiCurrent_next_indicator = (pucPackageBuffer[5 + iPayloadPosition] << 7) >> 7;
    pstSectionHead->uiSection_number = pucPackageBuffer[6 + iPayloadPosition];
    pstSectionHead->uiLast_section_number = pucPackageBuffer[7 + iPayloadPosition];
}

/******************************************
 *
 * 从PAT中获取PMT的前提信息
 *
 ******************************************/

void GetPAT_Info(TS_PAT_T *pstTS_PAT, int iPAT_ProgramCount, PAT_INFO_T *pstPAT_Info, int *iInfoCount) {
    int iLoopTime = 0;

    for (iLoopTime = 0; iLoopTime < iPAT_ProgramCount; iLoopTime++) {
        pstPAT_Info[*iInfoCount].uiPMT_PID = pstTS_PAT->stPAT_Program[iLoopTime].uiProgram_map_PID;
        pstPAT_Info[*iInfoCount].uiProgramNumber = pstTS_PAT->stPAT_Program[iLoopTime].uiProgram_number;
        (*iInfoCount)++;
    }
}

/******************************************
 *
 * 重置PAT数据
 *
 ******************************************/
void CleanPAT_Info(PAT_INFO_T *pstPAT_Info, int *piInfoCount) {
    *piInfoCount = 0;
    memset(pstPAT_Info, 0, sizeof(TS_PAT_PROGRAM_T) * PAT_PROGARM_MAX);
}


/******************************************************************
 * 将SectionBuffer的PAT头部信息存入TS_PAT中
 ******************************************************************/
void ParsePAT_SectionHead(TS_PAT_T *pstTS_PAT, unsigned char *pucSectionBuffer) {
    int iPAT_Length = 0;

    pstTS_PAT->uiTable_id = pucSectionBuffer[0];
    pstTS_PAT->uiSection_syntax_indicator = pucSectionBuffer[1] >> 7;
    pstTS_PAT->uiZero = (pucSectionBuffer[1] >> 6) & 0x1;
    pstTS_PAT->uiReserved_first = (pucSectionBuffer[1] >> 4) & 0x3;
    pstTS_PAT->uiSection_length = ((pucSectionBuffer[1] & 0x0f) << 8) | pucSectionBuffer[2];
    pstTS_PAT->uiTransport_stream_id = (pucSectionBuffer[3] << 8) | pucSectionBuffer[4];
    pstTS_PAT->uiReserved_second = pucSectionBuffer[5] >> 6;
    pstTS_PAT->uiVersion_number = (pucSectionBuffer[5] >> 1) & 0x1f;
    pstTS_PAT->uiCurrent_next_indicator = (pucSectionBuffer[5] << 7) >> 7;
    pstTS_PAT->uiSection_number = pucSectionBuffer[6];
    pstTS_PAT->uiLast_section_number = pucSectionBuffer[7];
    iPAT_Length = 3 + pstTS_PAT->uiSection_length;
    pstTS_PAT->uiCRC_32 = (pucSectionBuffer[iPAT_Length - 4] << 24) | (pucSectionBuffer[iPAT_Length - 3] << 16) |
                          (pucSectionBuffer[iPAT_Length - 2] << 8) | (pucSectionBuffer[iPAT_Length - 1]);
}

/******************************************************************
 * 解析PAT的关键信息，并将解析到的PAT头部信息与其拼接成完整的PAT表
 ******************************************************************/
int ParsePAT_Section(TS_PAT_T *pstTS_PAT, unsigned char *pucSectionBuffer) {
    int iPAT_Length = 0;
    int iPATProgramPosition = 8;
    int iPAT_ProgramCount = 0;

    memset(pstTS_PAT, 0, sizeof(TS_PAT_T));
    ParsePAT_SectionHead(pstTS_PAT, pucSectionBuffer);
    iPAT_Length = 3 + pstTS_PAT->uiSection_length;
    for (iPATProgramPosition = 8; iPATProgramPosition < iPAT_Length - 4; iPATProgramPosition += 4) {
        if (0x00 == ((pucSectionBuffer[iPATProgramPosition] << 8) | pucSectionBuffer[1 + iPATProgramPosition])) {
            pstTS_PAT->uiNetwork_PID = ((pucSectionBuffer[2 + iPATProgramPosition] & 0x1f) << 8) |
                                       pucSectionBuffer[3 + iPATProgramPosition];
            printf("*********************The network_PID is 0x%02x*******************\n",
                   pstTS_PAT->uiNetwork_PID);
        } else {
            pstTS_PAT->stPAT_Program[iPAT_ProgramCount].uiProgram_number =
                    (pucSectionBuffer[iPATProgramPosition] << 8) | pucSectionBuffer[1 + iPATProgramPosition];
            pstTS_PAT->stPAT_Program[iPAT_ProgramCount].uiReserved_third =
                    pucSectionBuffer[2 + iPATProgramPosition] >> 5;
            pstTS_PAT->stPAT_Program[iPAT_ProgramCount].uiProgram_map_PID =
                    ((pucSectionBuffer[2 + iPATProgramPosition] & 0x1f) << 8) |
                    pucSectionBuffer[3 + iPATProgramPosition];
            iPAT_ProgramCount++;
        }
    }
    return iPAT_ProgramCount;
}

/*********************************************
 * 解析TS中有效负载的开始位置
 * 获取传送流分组层中调整控制字段的值
 * 如果为0x00、0x10，即没有有效负载返回-1；
 * 如果为0x01、0x11，返回有效负载的开始位置
 *********************************************/
int GetTheLoadBeginPostion(unsigned char *pucPackageBuffer) {
    int iLoadBeginPostion = -1;
    TS_PACKAGE_HEAD_T stTS_PackageHead = {0};

    ParseTS_PackageHead(&stTS_PackageHead, pucPackageBuffer);
    switch (stTS_PackageHead.uiAdapptation_field_control) {
        case 0:
            return -1;
            break;
        case 1:
            iLoadBeginPostion = 4;
            printf("case 1: 4\n");
            break;
        case 2:
            return -1;
            break;
        case 3:
            iLoadBeginPostion = 4 + 1 + pucPackageBuffer[4];
            printf("case 3: %d\n", iLoadBeginPostion);
            break;
    }
    if (stTS_PackageHead.uiPayload_unit_start_indicator) {
        iLoadBeginPostion = iLoadBeginPostion + pucPackageBuffer[iLoadBeginPostion] + 1;
    }
    return iLoadBeginPostion;
}

/*********************************************
 * 判断一个Section是否结束
 *********************************************/
static int IsOneSectionOver(SECTION_HEAD_T stSectionHead, int iAddLength) {
    int iSectionLength = (int) (stSectionHead.uiSection_Length + 3);
    if (iAddLength == iSectionLength) {
        return 1;
    }
    return 0;
}

/*********************************************
 * 解析Section版本数据
 *********************************************/
static int IsVersionChange(SECTION_HEAD_T *pstSectionHead, unsigned int *puiRecordVersion) {
    if (INITIAL_VERSION == (*puiRecordVersion)) {
        (*puiRecordVersion) = pstSectionHead->uiVersion_number;
    }
    if ((*puiRecordVersion) != pstSectionHead->uiVersion_number) {
        return 1;
    }
    return 0;
}

/*********************************************
 * 把缓存数据复制到Section中
 *********************************************/
static void
PutTheBufferToSection(unsigned char *pucSectionBuffer, unsigned char *pucPackageBuffer, int *piAlreadyAddSection,
                      SECTION_HEAD_T *pstPAT_SectionHead, int iLoadBeginPosition) {
    int iCopyLength = 0;

    iCopyLength = 188 - iLoadBeginPosition;
    if ((int) (pstPAT_SectionHead->uiSection_Length + 3) < (188 - iLoadBeginPosition + (*piAlreadyAddSection))) {
        iCopyLength = pstPAT_SectionHead->uiSection_Length + 3 - (*piAlreadyAddSection);
    }
    memcpy(pucSectionBuffer + (*piAlreadyAddSection), pucPackageBuffer + iLoadBeginPosition, iCopyLength);
    (*piAlreadyAddSection) = (*piAlreadyAddSection) + iCopyLength;
}

/*********************************************
 *
 * 解析TS的一个Section
 * -1：缓存流中没有Section
 * 0：版本变化了
 * 1：获取到正确的Section
 * 2:数据是错误的Section
 *
 *********************************************/
int GetOneSection(FILE *pfTsFile, int iTsLength, unsigned char *pucSectionBuffer, unsigned int uiPID,
                  unsigned int uiTableId, unsigned int *puiVersion) {
    int iPayloadPosition = -1;
    int iFlagSectionStart = 0;
    int iAlreadyAddSection = 0;
    int iLoadBeginPosition = 0;
    unsigned char ucPackageBuffer[204] = {0};
    TS_PACKAGE_HEAD_T stTS_PackageHead = {0};
    SECTION_HEAD_T stSectionHead = {0};
    memset(pucSectionBuffer, 0, sizeof(char) * SECTION_MAX_LENGTH_4096);
    while (!feof(pfTsFile)) {
        if ((sizeof(char) * iTsLength) !=
            (fread((unsigned char *) ucPackageBuffer, sizeof(char), iTsLength, pfTsFile))) {
            break;
        }
        ParseTS_PackageHead(&stTS_PackageHead, ucPackageBuffer);

        //输出TS_HEAD信息
        PrintTS_PES_HEAD(stTS_PackageHead);
        //检查head标志
        if (stTS_PackageHead.uiTransport_error_indicator == 1) {
            printf("uiTransport_error_indicator is 1\n");
            continue;
        }

        if ((stTS_PackageHead.uiPID == uiPID) && (0x47 == stTS_PackageHead.uiSync_byte)) {

            iPayloadPosition = GetTheLoadBeginPostion(ucPackageBuffer);
            if (-1 == iPayloadPosition) /* 没有有效负载 */
            {
                continue;
            }
            // find 0x00 from payload
            if ((1 == stTS_PackageHead.uiPayload_unit_start_indicator) &&
                ((ucPackageBuffer[iPayloadPosition] == uiTableId) || (DEFAULT_TABLE_ID == uiTableId)) &&
                (1 != iFlagSectionStart)) {
                ParsePATSectionHeader(&stSectionHead, ucPackageBuffer);

                if (1 == IsVersionChange(&stSectionHead, puiVersion)) {
                    return 0; /* version number change */
                }
                iFlagSectionStart = 1;
                iLoadBeginPosition = GetTheLoadBeginPostion(ucPackageBuffer);
                PutTheBufferToSection(pucSectionBuffer, ucPackageBuffer, &iAlreadyAddSection, &stSectionHead,
                                      iLoadBeginPosition);
            } else {   // 跨包数据处理
                if (1 == iFlagSectionStart) {
                    if (1 == stTS_PackageHead.uiPayload_unit_start_indicator) {
                        PutTheBufferToSection(pucSectionBuffer, ucPackageBuffer, &iAlreadyAddSection, &stSectionHead,
                                              4 + 1);
                    } else {
                        PutTheBufferToSection(pucSectionBuffer, ucPackageBuffer, &iAlreadyAddSection, &stSectionHead,
                                              4);
                    }
                }
            }
        }
//        if (1 == iFlagSectionStart) {
//            if (1 == IsOneSectionOver(stSectionHead, iAlreadyAddSection)) {
//                return 1;
//
//                if (1 == isHasCRC32_TableId(uiTableId))    // 如果属于需要CRC32校验的表格 先进行CRC32校验 通过TABLEID判断
//                {
//                    if (1 == Verify_CRC_32(pucSectionBuffer)) {
//                        return 1;
//                    } else {
//                        printf("Verify_CRC_32 a CRC error \n");
//                        return 2;
//                    }
//                } else {
//                    return 1;
//                }
//            }
//        }
    }
    return -1;
}

int GetOneSectionByPID(FILE *pfTsFile, int iTsLength, unsigned char *pucSectionBuffer, unsigned int uiPID,
                       unsigned int *puiVersion) {
    unsigned int uiVersion = INITIAL_VERSION;
    return GetOneSection(pfTsFile, iTsLength, pucSectionBuffer, uiPID, DEFAULT_TABLE_ID, &uiVersion);
}
/******************************************
 *
 * 判断是否已获取
 *
 ******************************************/
int IsAllSectionOver(unsigned char *pucSectionBuffer, unsigned int *puiRecordSectionNumber)
{
    unsigned int uiSectionCount = puiRecordSectionNumber[0];
    unsigned int uiLastSectionNumber = pucSectionBuffer[7];

    if (uiSectionCount == (uiLastSectionNumber + 1))
    {
        return 1;
    }
    return 0;
}


int JudgmentPackageTenTimes(FILE *pfTsFile, int iTsPosition, int iTsLength) {
    int iFirstPackageByte = 0;

    if (-1 == fseek(pfTsFile, iTsPosition + 1, SEEK_SET)) {
        return -1;
    }
    if (-1 == fseek(pfTsFile, iTsLength - 1, SEEK_CUR)) {
        return -1;
    }

    if (feof(pfTsFile)) {
        return -1;
    }
    iFirstPackageByte = fgetc(pfTsFile);
    if (0x47 != iFirstPackageByte) {
        return -1;
    }
    return iTsLength;
}

int ParseTsLength(FILE *pfTsFile, int *piTsPosition) {
    int iFirstPackageByte = 0;
    while (!feof(pfTsFile)) {
        iFirstPackageByte = fgetc(pfTsFile);
        if (0x47 == iFirstPackageByte) {
            // packet length 188
            if (188 == JudgmentPackageTenTimes(pfTsFile, *piTsPosition, 188)) {
                return 188;
            }
            // packet length 204
            if (204 == JudgmentPackageTenTimes(pfTsFile, *piTsPosition, 204)) {
                return 204;
            }
        }
        (*piTsPosition)++;

        if (-1 == fseek(pfTsFile, *piTsPosition, SEEK_SET)) {
            printf("The file error\n");
            return -1;
        }
    }
    printf("The file is not the transport stream\n");
    return -1;
}

int ParsePAT_Table(FILE *pfTsFile, int iTsPosition, int iTsLength, PAT_INFO_T *pstPAT_Info_T) {
    printf("\n=================================ParsePAT_Table Start================================= \n");
    int iTemp = 0;
    int iInfoCount = 0;
    int iPATProgramCount = 0;
    TS_PAT_T stTS_PAT_T = {0};
    unsigned int uiVersion = INITIAL_VERSION;
    unsigned int uiRecordSectionNumber[SECTION_COUNT_256] = {0};
    unsigned char ucSectionBuffer[SECTION_MAX_LENGTH_4096] = {0};

    if (-1 == fseek(pfTsFile, iTsPosition, SEEK_SET)) {
        printf("Parse table error\n");
        return -1;
    }

    while (!feof(pfTsFile)) {
        iTemp = GetOneSection(pfTsFile, iTsLength, ucSectionBuffer, PAT_PID, PAT_TABLE_ID, &uiVersion);
        switch (iTemp) {
            case 0:
                printf("Enter if (0 == iTemp) in PARSE_PAT\n");
                uiVersion = INITIAL_VERSION;
                memset(uiRecordSectionNumber, 0, sizeof(char) * SECTION_COUNT_256);
                fseek(pfTsFile, 0 - iTsLength, SEEK_CUR);
                CleanPAT_Info(pstPAT_Info_T, &iInfoCount);
                break;
            case 1:
                printf("Enter if (1 == iTemp) in PARSE_PAT\n");
                if (0 == IsSectionGetBefore(ucSectionBuffer, uiRecordSectionNumber)) {
                    printf("Enter if (0 == IsSectionGetBefore) in PARSE_PAT\n");
                    iPATProgramCount = ParsePAT_Section(&stTS_PAT_T, ucSectionBuffer);

                    GetPAT_Info(&stTS_PAT_T, iPATProgramCount, pstPAT_Info_T, &iInfoCount);
                    if (1 == PRINTFPAT_INFO) {
                        PrintPAT(&stTS_PAT_T, iPATProgramCount);
                    }
                }
                if (1 == IsAllSectionOver(ucSectionBuffer, uiRecordSectionNumber)) {
                    printf("Enter if (1 == IsAllSectionOver) in PARSE_PAT\n");
                    printf("return iInfoCount, iInfoCount is: %d\n", iInfoCount);
                    printf("\n\n=================================ParsePAT_Table End=================================== \n\n");
                    return iInfoCount;
                }
                break;
            case 2:
                break;
            case -1:
                printf("Enter if (-1 == iTemp) in PARSE_PAT\n");
                printf("return iInfoCount, iInfoCount is: %d\n", iInfoCount);
                printf("\n\n=================================ParsePAT_Table End=================================== \n\n");
                return iInfoCount;
                break;
            default:
                printf("ParsePAT_Table switch (iTemp) default\n");
                break;
        }
    }
    printf("return 0\n");
    printf("\n=================================ParsePAT_Table End===================================\n\n");
    return 0;
}

int main(int argc, char **argvs) {
    // argvs[0]
    int iPosT = 0;
    char *file = argvs[1];
    fprintf(stdout, "Parse file: %s\n", file);
    FILE *ifile = fopen(file, "rb+");
    if (ifile == NULL) {
        fprintf(stdout, "Failed to open files!\n");
        return -1;
    }

    int ts_len = ParseTsLength(ifile, &iPosT);
    if (iPosT != 0) {
        fprintf(stderr, "Ts file is not valid");
        return -1;
    }
    fprintf(stdout, "Ts header Length is %d\n", ts_len);

    PAT_INFO_T stPAT_Info[PROGRAM_MAX] = {0};
    PMT_INFO_T stPMT_Info[PROGRAM_MAX] = {0};
    CAT_INFO_T stCAT_Info[CA_SYSTEM_MAX] = {0};
    // parse PAT
    int iProgramCount = ParsePAT_Table(ifile, iPosT, ts_len, stPAT_Info);

    printf("Program count is %d\n", iProgramCount);
    return 0;
}